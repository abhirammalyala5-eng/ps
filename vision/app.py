"""Food image analysis API: FoodSeg103 segmentation + Ollama Qwen3-VL."""

from __future__ import annotations

import base64
import cv2
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
import segmentation_models_pytorch as smp


ROOT = Path(__file__).resolve().parent
SEGMENTER_PATH = Path(os.getenv("FOODSEG_MODEL", ROOT / "models/FoodSeg103/best_model.pth"))
LABELS_PATH = Path(os.getenv("FOODSEG_LABELS", ROOT / "models/FoodSeg103/labels_config.json"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl:2b")
QWEN_NUM_CTX = int(os.getenv("QWEN_NUM_CTX", "16384"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(40 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
CONFIDENCE = float(os.getenv("FOODSEG_CONFIDENCE", "0.25"))
FOODSEG_TILE_SIZE = int(os.getenv("FOODSEG_TILE_SIZE", "512"))
FOODSEG_TILE_OVERLAP = int(os.getenv("FOODSEG_TILE_OVERLAP", "64"))

_ollama_endpoint = urlparse(OLLAMA_URL)
if _ollama_endpoint.scheme not in {"http", "https"} or _ollama_endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
    raise RuntimeError("Offline mode requires OLLAMA_URL to point to this computer (localhost)")
if QWEN_MODEL.lower().endswith("-cloud"):
    raise RuntimeError("Offline mode cannot use an Ollama cloud model; configure a locally installed model")

if not SEGMENTER_PATH.is_file():
    raise RuntimeError(f"FoodSeg103 weights not found: {SEGMENTER_PATH}")
if not LABELS_PATH.is_file():
    raise RuntimeError(f"FoodSeg103 class labels not found: {LABELS_PATH}")

with LABELS_PATH.open(encoding="utf-8") as f:
    _label_config = json.load(f)
CLASS_NAMES = {int(key): value for key, value in _label_config["id2label"].items()}
if len(CLASS_NAMES) != 104:
    raise RuntimeError(f"Expected 104 FoodSeg103 classes, found {len(CLASS_NAMES)}")

segmenter = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=104
)
segmenter.load_state_dict(torch.load(SEGMENTER_PATH, map_location="cpu", weights_only=True))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmenter.to(DEVICE).eval()
app = FastAPI(title="Food Vision API", version="1.0.0")
logger = logging.getLogger("foodvision")
if FOODSEG_TILE_SIZE < 32 or FOODSEG_TILE_SIZE % 32 != 0:
    raise RuntimeError("FOODSEG_TILE_SIZE must be a positive multiple of 32")
if not 0 <= FOODSEG_TILE_OVERLAP < FOODSEG_TILE_SIZE:
    raise RuntimeError("FOODSEG_TILE_OVERLAP must be smaller than FOODSEG_TILE_SIZE")
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "dish": {"type": "string"},
        "components": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "food_area_fraction": {"type": "number"},
                "height_fraction_min": {
                    "type": "number",
                    "maximum": 0.2,
                    "description": "Minimum average food height divided by plate diameter; 0.1 means one tenth of the plate diameter, not 0.1 cm. Typical plated food is 0.05 to 0.2.",
                },
                "height_fraction_max": {
                    "type": "number",
                    "maximum": 0.2,
                    "description": "Maximum average food height divided by plate diameter; 0.1 means one tenth of the plate diameter, not 0.1 cm. Typical plated food is 0.05 to 0.2.",
                },
                "shape_factor": {"type": "number"},
                "volume_ml_min": {"type": "number"},
                "volume_ml_max": {"type": "number"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["name", "food_area_fraction", "height_fraction_min", "height_fraction_max", "shape_factor", "volume_ml_min", "volume_ml_max", "confidence"],
            "additionalProperties": False,
        }},
    },
    "required": ["dish", "components"],
    "additionalProperties": False,
}

DENSITY_TABLE = json.loads((ROOT / "food_density.json").read_text(encoding="utf-8"))


def segmentation_summary(image: Image.Image) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Segment every original-resolution pixel using overlapping, non-resized tiles."""
    width, height = image.size
    pixels = np.asarray(image, dtype=np.uint8)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tile_size = FOODSEG_TILE_SIZE
    overlap = FOODSEG_TILE_OVERLAP

    def tile_starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def blend_weights(start: int, valid_length: int, full_length: int) -> np.ndarray:
        weights = np.ones(valid_length, dtype=np.float32)
        ramp_length = min(overlap, valid_length)
        if ramp_length and start > 0:
            weights[:ramp_length] = np.linspace(0.05, 1.0, ramp_length, dtype=np.float32)
        if ramp_length and start + valid_length < full_length:
            weights[-ramp_length:] = np.minimum(
                weights[-ramp_length:],
                np.linspace(1.0, 0.05, ramp_length, dtype=np.float32),
            )
        return weights

    pred = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    best_score = np.full((height, width), -1.0, dtype=np.float32)

    for y in tile_starts(height):
        valid_height = min(tile_size, height - y)
        input_height = max(32, ((valid_height + 31) // 32) * 32)
        y_weights = blend_weights(y, valid_height, height)
        for x in tile_starts(width):
            valid_width = min(tile_size, width - x)
            input_width = max(32, ((valid_width + 31) // 32) * 32)
            x_weights = blend_weights(x, valid_width, width)

            tile = pixels[y:y + valid_height, x:x + valid_width].astype(np.float32) / 255.0
            normalized = (tile - mean) / std
            tensor_array = np.zeros((1, 3, input_height, input_width), dtype=np.float32)
            tensor_array[0, :, :valid_height, :valid_width] = normalized.transpose(2, 0, 1)
            tensor = torch.from_numpy(tensor_array).to(DEVICE)

            with torch.inference_mode():
                logits = segmenter(tensor)
                probabilities = torch.softmax(logits, dim=1)[0, :, :valid_height, :valid_width]
                confidence, prediction = probabilities.max(dim=0)

            tile_pred = prediction.cpu().numpy().astype(np.uint8)
            tile_conf = confidence.cpu().numpy().astype(np.float32)
            tile_weight = y_weights[:, None] * x_weights[None, :]
            score = tile_conf * tile_weight
            score_region = best_score[y:y + valid_height, x:x + valid_width]
            replace = score > score_region
            pred_region = pred[y:y + valid_height, x:x + valid_width]
            confidence_region = confidence_map[y:y + valid_height, x:x + valid_width]
            pred_region[replace] = tile_pred[replace]
            confidence_region[replace] = tile_conf[replace]
            score_region[replace] = score[replace]

    areas: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    total_pixels = width * height
    food_pixels = 0
    for class_id in np.unique(pred):
        class_id = int(class_id)
        if class_id == 0 or class_id not in CLASS_NAMES:
            continue
        mask = pred == class_id
        count = int(mask.sum())
        fraction = count / total_pixels
        mean_conf = float(confidence_map[mask].mean())
        if fraction < 0.001 or mean_conf < CONFIDENCE:
            continue
        class_name = CLASS_NAMES[class_id]
        food_pixels += count
        areas.append({"class": class_name, "area_percent": round(fraction * 100, 2)})
        detections.append({"class": class_name, "confidence": round(mean_conf, 4)})
    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return areas, detections, food_pixels


def detect_top_plate_scale(image: Image.Image, plate_diameter_cm: float, food_pixels: int) -> dict[str, Any]:
    """Approximate top-view pixel scale from a fully visible plate rim."""
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    short_side = min(image.size)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=short_side * 0.45,
        param1=100, param2=30, minRadius=int(short_side * 0.28),
        maxRadius=int(short_side * 0.58),
    )
    if circles is None:
        return {"detected": False, "confidence": "low", "reason": "plate rim not detected"}

    h, w = gray.shape
    cx0, cy0 = w / 2, h / 2
    candidates = []
    for x, y, radius in circles[0]:
        # Calibration is only reported when the full circular boundary is in-frame.
        if x - radius < 2 or y - radius < 2 or x + radius > w - 2 or y + radius > h - 2:
            continue
        center_offset = np.hypot((x - cx0) / w, (y - cy0) / h)
        candidates.append((center_offset, -radius, float(x), float(y), float(radius)))
    if not candidates:
        return {"detected": False, "confidence": "low", "reason": "plate appears cropped"}

    offset, _, cx, cy, radius = min(candidates)
    diameter_px = radius * 2
    pixels_per_cm = diameter_px / plate_diameter_cm
    plate_area_px = np.pi * radius * radius
    food_to_plate_ratio = food_pixels / plate_area_px
    if food_to_plate_ratio > 0.95:
        return {"detected": False, "confidence": "low", "reason": "candidate plate circle is inconsistent with food-mask area"}
    footprint_cm2 = food_to_plate_ratio * (np.pi * (plate_diameter_cm / 2) ** 2)
    confidence = "medium" if offset < 0.12 else "low"
    return {
        "detected": True,
        "confidence": confidence,
        "plate_diameter_px": round(diameter_px, 1),
        "pixels_per_cm": round(pixels_per_cm, 2),
        "food_footprint_cm2": round(footprint_cm2, 1),
        "method": "Hough circle estimate from top-view plate rim",
    }

def parse_component_response(text: str, plate_diameter_cm: float) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Qwen did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("components"), list):
        raise ValueError('Expected JSON with a "components" list')
    components = []
    for item in value["components"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            area_share = max(0.0, min(float(item["food_area_fraction"]), 1.0))
            # Qwen tends to emit 0.1 for a pile around one tenth of the plate diameter.
            # Treat height as a dimensionless ratio, then apply the user's physical plate size.
            # Cap average plated-food height at 20% of plate diameter (4 cm on a 20 cm plate).
            # Larger fractions from this small VLM were producing implausibly huge servings.
            height_fraction_min = max(0.0, min(float(item["height_fraction_min"]), 0.2))
            height_fraction_max = max(height_fraction_min, min(float(item["height_fraction_max"]), 0.2))
            height_min = height_fraction_min * plate_diameter_cm
            height_max = height_fraction_max * plate_diameter_cm
            shape_factor = max(0.2, min(float(item["shape_factor"]), 1.0))
            low = max(0.0, min(float(item["volume_ml_min"]), 5000.0))
            high = max(low, min(float(item["volume_ml_max"]), 5000.0))
        except (KeyError, TypeError, ValueError):
            continue
        confidence = item.get("confidence", "low")
        if confidence not in ("low", "medium", "high"):
            confidence = "low"
        components.append({"name": item["name"], "food_area_fraction": area_share,
                           "height_cm_min": height_min, "height_cm_max": height_max,
                           "shape_factor": shape_factor, "volume_ml_min": low,
                           "volume_ml_max": high, "confidence": confidence})
    return {"dish": str(value.get("dish", "food")), "components": components}


def density_category(name: str) -> str:
    name = name.lower()
    if any(x in name for x in ("rice", "noodle", "pasta", "grain", "biryani", "pilaf")):
        return "cooked_grain"
    if any(x in name for x in ("chicken", "beef", "pork", "lamb", "fish", "meat", "shrimp", "prawn")):
        return "cooked_meat"
    if any(x in name for x in ("dal", "lentil", "bean", "chickpea", "curry", "stew")):
        return "legume_or_curry"
    if any(x in name for x in ("sauce", "gravy", "dressing", "soup")):
        return "sauce"
    if any(x in name for x in ("bread", "roti", "chapati", "naan", "flatbread")):
        return "bread"
    if any(x in name for x in ("cheese", "paneer", "yogurt", "curd")):
        return "dairy"
    if any(x in name for x in ("fruit", "apple", "banana", "mango", "orange")):
        return "fruit"
    if any(x in name for x in ("vegetable", "onion", "pea", "carrot", "potato", "herb", "salad", "cilantro")):
        return "vegetable"
    return "mixed_food"


def build_weight_estimates(components: list[dict[str, Any]], plate_scale: dict[str, Any]) -> list[dict[str, Any]]:
    estimates = []
    total_share = sum(item["food_area_fraction"] for item in components)
    for item in components:
        category = density_category(item["name"])
        density = DENSITY_TABLE[category]
        if plate_scale.get("detected") and total_share > 0:
            normalized_share = item["food_area_fraction"] / total_share
            area_cm2 = plate_scale["food_footprint_cm2"] * normalized_share
            shape = item["shape_factor"]
            vol_min = area_cm2 * item["height_cm_min"] * max(0.2, shape * 0.8)
            vol_max = area_cm2 * item["height_cm_max"] * min(1.0, shape * 1.2)
            method = "calibrated top-view food footprint × estimated side-view height × shape factor"
        else:
            vol_min, vol_max = item["volume_ml_min"], item["volume_ml_max"]
            method = "Qwen two-view volume estimate; plate calibration unavailable"
        grams_min = vol_min * density["min"]
        grams_max = vol_max * density["max"]
        estimate = ((vol_min + vol_max) / 2) * ((density["min"] + density["max"]) / 2)
        confidence = item["confidence"]
        if not plate_scale.get("detected"):
            confidence = "low"
        elif plate_scale.get("confidence") == "low" and confidence == "high":
            confidence = "medium"
        estimates.append({
            "name": item["name"],
            "volume_ml": {"min": round(vol_min), "estimate": round((vol_min + vol_max) / 2), "max": round(vol_max)},
            "density_g_per_ml": {"min": density["min"], "max": density["max"]},
            "weight_g": {"min": round(grams_min / 10) * 10, "estimate": round(estimate / 10) * 10, "max": round(grams_max / 10) * 10},
            "confidence": confidence,
            "method": method + " × food-category density range",
        })
    return estimates


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "segmenter": "loaded",
        "segmenter_name": "FoodSeg103",
        "segmenter_architecture": "DeepLabV3+ with MobileNetV2 encoder",
        "segmenter_class_count": len(CLASS_NAMES),
        "segmenter_classes": list(CLASS_NAMES.values()),
        "segmenter_device": str(DEVICE),
        "segmenter_input_mode": "native-resolution overlapping tiles",
        "segmenter_tile_size": FOODSEG_TILE_SIZE,
        "segmenter_tile_overlap": FOODSEG_TILE_OVERLAP,
        "qwen_context_tokens": QWEN_NUM_CTX,
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "density_categories": list(DENSITY_TABLE),
        "qwen_model": QWEN_MODEL,
        "offline_runtime": True,
    }


async def _read_food_image(upload: UploadFile, field_name: str) -> tuple[Image.Image, bytes]:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{field_name} must be an image")
    raw = await upload.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field_name} exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        decoded = Image.open(io.BytesIO(raw))
        if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"{field_name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit; the API will not downscale it",
            )
        image = ImageOps.exif_transpose(decoded).convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit; the API will not downscale it",
        )
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail=f"Could not decode {field_name}")
    return image, raw


@app.post("/vision")
async def vision(
    top_image: UploadFile = File(..., description="Photo taken directly above the meal"),
    side_image: UploadFile = File(..., description="Photo taken from the side at plate level"),
    plate_diameter_cm: float = Form(..., description="User-selected plate diameter in cm (typical choices: 20, 25, or 30)"),
) -> dict[str, Any]:
    """Estimate ingredient weights from paired views and a known plate diameter."""
    if not 15 <= plate_diameter_cm <= 40:
        raise HTTPException(status_code=422, detail="plate_diameter_cm must be between 15 and 40 cm")
    top, top_original_bytes = await _read_food_image(top_image, "top_image")
    side, side_original_bytes = await _read_food_image(side_image, "side_image")

    started = time.perf_counter()
    top_areas, top_detections, top_food_pixels = segmentation_summary(top)
    side_areas, side_detections, _ = segmentation_summary(side)
    plate_scale = detect_top_plate_scale(top, plate_diameter_cm, top_food_pixels)
    segmented_at = time.perf_counter()
    prompt = (
        "You are given TWO photos of the SAME serving: image 1 is the top-down view; image 2 is the "
        "side view at plate level. Use the top view to estimate footprint and the side view to estimate "
        "food height/volume. Cross-check the two views so you do not count the serving twice. "
        "Carefully inspect the ENTIRE meal yourself. "
        f"The user selected a plate diameter of {plate_diameter_cm:g} cm. Assume the visible plate is "
        "that size and use its rim as a real-world scale reference in both photos; estimate the food "
        "footprint and height relative to the plate rather than treating image pixels as centimeters. "
        f"Computer-vision scale check from the top view: {json.dumps(plate_scale)}. "
        "If this scale check says not detected or conflicts with what you see, lower your confidence. "
        "The photos are the source of truth and have "
        "higher priority than the segmentation hints. First identify the dish, then make an exhaustive "
        "ingredient inventory of substantial visible food components. Do not invent hidden spices or "
        "recipe ingredients that cannot be seen. For each component, estimate its share of the total "
        "segmented food footprint (all component shares should sum to about 1), its average height range "
        "as a fraction of the selected plate diameter (not centimeters): for a 20 cm plate, a food height "
        "one tenth as tall as the plate diameter is 0.10. The server converts this ratio to centimeters. "
        "Do not return 0.10 when you mean 0.10 cm; the field is a dimensionless ratio. Estimate a shape factor from "
        "0.2 to 1.0 (flat/irregular foods lower, compact mounds closer to 1). A typical plated layer is "
        "0.05–0.20 of the plate diameter. Also provide a fallback "
        "low/high volume in milliliters in case computer-vision plate calibration fails. Use recipe "
        "knowledge only to identify visible foods. Return a confidence of low, medium, or high. "
        "Food segmentation returned these OPTIONAL, noisy pixel-area "
        "hints: top view="
        f"{json.dumps(top_areas, ensure_ascii=False)}, side view={json.dumps(side_areas, ensure_ascii=False)}. "
        "The semantic segmenter can miss or confuse ingredients, especially for small or occluded regions. "
        "Use its class labels as useful but imperfect evidence; cross-check against the photos and do not "
        "let one uncertain label dominate your visual analysis. "
        "and ignore hints that conflict with the photos. If plate calibration succeeds, the server will "
        "calculate volume as calibrated food footprint × your component area share × side-view height "
        "× shape factor; otherwise it uses your fallback volume range. The server then applies a "
        "food-category density range to produce gram ranges. Avoid double-counting views. Do not return "
        "gram weights. "
        'Return only JSON with keys "dish" and "components"; each component must have "name", '
        '"food_area_fraction", "height_fraction_min", "height_fraction_max", "shape_factor", "volume_ml_min", '
        '"volume_ml_max", and "confidence".'
    )
    payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(top_original_bytes).decode("ascii"),
                       base64.b64encode(side_original_bytes).decode("ascii")],
        }],
        "stream": False,
        "think": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 768, "num_ctx": QWEN_NUM_CTX},
        "keep_alive": "10m",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as client:
            response = await client.post(f"{OLLAMA_URL.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
        ollama_result = response.json()
        message = ollama_result.get("message", {})
        answer = message.get("content") or message.get("thinking", "")
        try:
            parsed = parse_component_response(answer, plate_diameter_cm)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Qwen returned unusable content=%r thinking=%r", answer[:300], ollama_result.get("message", {}).get("thinking", "")[:200])
            raise ValueError(f"{exc}; response={answer[:160]!r}") from exc
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama. Start it with `ollama serve`.")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        if exc.response.status_code == 404:
            detail = (
                f"Local Ollama model '{QWEN_MODEL}' is unavailable. Verify it with `ollama list`; "
                "the API does not download models."
            )
        elif "exceed_context_size_error" in detail or "exceeds the available context size" in detail:
            detail = (
                f"The request exceeded Ollama's context window. The API is configured for "
                f"{QWEN_NUM_CTX} tokens; set QWEN_NUM_CTX higher and restart the API."
            )
        raise HTTPException(status_code=502, detail=detail)
    except (httpx.TimeoutException, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Qwen response error: {exc}")

    logger.info(
        "vision completed: segmentation=%.2fs qwen=%.2fs total=%.2fs top=%sx%s side=%sx%s",
        segmented_at - started, time.perf_counter() - segmented_at, time.perf_counter() - started,
        top.width, top.height, side.width, side.height,
    )
    estimates = build_weight_estimates(parsed["components"], plate_scale)
    top_classes = [item["class"] for item in top_areas]
    side_classes = [item["class"] for item in side_areas]
    views_share_class = bool(set(top_classes) & set(side_classes))
    limitations = [
        "These are AI-assisted estimates, not measured weights.",
        "Plate calibration depends on a visible, fully framed plate rim; occlusion or perspective can reduce accuracy.",
        "The multi-class segmentation model can miss or confuse ingredients, especially when foods overlap.",
        "Food density varies with packing, moisture, cooking, and recipe.",
    ]
    if top_classes and side_classes and not views_share_class:
        limitations.append(
            "FoodSeg labels differ between the two views (top: " + ", ".join(top_classes[:3]) +
            "; side: " + ", ".join(side_classes[:3]) +
            "). Check that both photos show the same serving; the segmenter can also mislabel food."
        )
        for estimate in estimates:
            if estimate["confidence"] == "high":
                estimate["confidence"] = "medium"
    ingredient_totals = {item["name"]: item["weight_g"]["estimate"] for item in estimates}
    return {
        "dish": parsed["dish"],
        "ingredients": ingredient_totals,
        "estimates": estimates,
        "scale_reference": {"plate_diameter_cm": plate_diameter_cm, **plate_scale},
        "image_processing": {
            "top_image_px": {"width": top.width, "height": top.height},
            "side_image_px": {"width": side.width, "height": side.height},
            "qwen_input": "original uploaded file bytes; no API resize, crop, or re-encode",
            "segmentation_input": "native-resolution overlapping tiles; all original pixels are covered",
        },
        "method": "Qwen estimates component volumes from two views; backend multiplies volume ranges by food-category density ranges.",
        "limitations": limitations,
        "segmentation": {
        "top_view": {"areas": top_areas, "detections": top_detections},
        "side_view": {"areas": side_areas, "detections": side_detections},
        },
    }
