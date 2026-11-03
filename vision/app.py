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

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SEGMENTER_PATH = Path(os.getenv("FOODSEG_MODEL", ROOT / "models/FoodSeg103-YOLO11l/best.pt"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl:2b")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
CONFIDENCE = float(os.getenv("FOODSEG_CONFIDENCE", "0.25"))

if not SEGMENTER_PATH.is_file():
    raise RuntimeError(f"FoodSeg103 weights not found: {SEGMENTER_PATH}")

segmenter = YOLO(str(SEGMENTER_PATH))
app = FastAPI(title="Food Vision API", version="1.0.0")
logger = logging.getLogger("foodvision")
MAX_SIDE = int(os.getenv("VISION_MAX_SIDE", "768"))
SEGMENT_SIZE = int(os.getenv("FOODSEG_IMAGE_SIZE", "512"))
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "dish": {"type": "string"},
        "components": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "food_area_fraction": {"type": "number"},
                "height_cm_min": {"type": "number"},
                "height_cm_max": {"type": "number"},
                "shape_factor": {"type": "number"},
                "volume_ml_min": {"type": "number"},
                "volume_ml_max": {"type": "number"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["name", "food_area_fraction", "height_cm_min", "height_cm_max", "shape_factor", "volume_ml_min", "volume_ml_max", "confidence"],
            "additionalProperties": False,
        }},
    },
    "required": ["dish", "components"],
    "additionalProperties": False,
}

DENSITY_TABLE = json.loads((ROOT / "food_density.json").read_text(encoding="utf-8"))


def segmentation_summary(image: Image.Image) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return class area fractions, per-instance metadata, and union mask pixels."""
    result = segmenter.predict(
        image, conf=CONFIDENCE, imgsz=SEGMENT_SIZE, device=0 if __import__("torch").cuda.is_available() else "cpu",
        half=__import__("torch").cuda.is_available(), max_det=30, verbose=False,
    )[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return [], [], 0

    names = result.names
    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    h, w = image.height, image.width

    class_masks: dict[int, np.ndarray] = {}
    instances: list[dict[str, Any]] = []
    for mask, class_id, confidence in zip(masks, classes, confidences):
        binary = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(
            (w, h), Image.Resampling.NEAREST
        )
        binary_array = np.asarray(binary) > 0
        class_masks[class_id] = class_masks.get(class_id, np.zeros((h, w), dtype=bool)) | binary_array
        instances.append({
            "class": str(names.get(int(class_id), class_id)),
            "confidence": round(float(confidence), 4),
        })

    areas = []
    for class_id, mask in class_masks.items():
        fraction = float(mask.sum()) / float(h * w)
        if fraction > 0:
            areas.append({
                "class": str(names.get(int(class_id), class_id)),
                "area_percent": round(fraction * 100, 2),
            })
    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    union_pixels = int(np.logical_or.reduce(list(class_masks.values())).sum()) if class_masks else 0
    return areas, instances, union_pixels


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


def parse_component_response(text: str) -> dict[str, Any]:
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
            height_min = max(0.0, min(float(item["height_cm_min"]), 30.0))
            height_max = max(height_min, min(float(item["height_cm_max"]), 30.0))
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
        "segmenter_class_count": len(segmenter.names),
        "segmenter_classes": list(segmenter.names.values()),
        "density_categories": list(DENSITY_TABLE),
        "qwen_model": QWEN_MODEL,
    }


async def _read_food_image(upload: UploadFile, field_name: str) -> Image.Image:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{field_name} must be an image")
    raw = await upload.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field_name} exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail=f"Could not decode {field_name}")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return image


@app.post("/vision")
async def vision(
    top_image: UploadFile = File(..., description="Photo taken directly above the meal"),
    side_image: UploadFile = File(..., description="Photo taken from the side at plate level"),
    plate_diameter_cm: float = Form(..., description="User-selected plate diameter in cm (typical choices: 20, 25, or 30)"),
) -> dict[str, Any]:
    """Estimate ingredient weights from paired views and a known plate diameter."""
    if not 15 <= plate_diameter_cm <= 40:
        raise HTTPException(status_code=422, detail="plate_diameter_cm must be between 15 and 40 cm")
    top = await _read_food_image(top_image, "top_image")
    side = await _read_food_image(side_image, "side_image")

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
        "in centimeters from the side view using the plate diameter as scale, and a shape factor from "
        "0.2 to 1.0 (flat/irregular foods lower, compact mounds closer to 1). Also provide a fallback "
        "low/high volume in milliliters in case computer-vision plate calibration fails. Use recipe "
        "knowledge only to identify visible foods. Return a confidence of low, medium, or high. "
        "Food segmentation returned these OPTIONAL, noisy pixel-area "
        "hints: top view="
        f"{json.dumps(top_areas, ensure_ascii=False)}, side view={json.dumps(side_areas, ensure_ascii=False)}. "
        "The segmenter may say only 'food', omit ingredients, "
        "or be wrong. Do not copy its labels as the answer, do not let it limit your visual analysis, "
        "and ignore hints that conflict with the photos. If plate calibration succeeds, the server will "
        "calculate volume as calibrated food footprint × your component area share × side-view height "
        "× shape factor; otherwise it uses your fallback volume range. The server then applies a "
        "food-category density range to produce gram ranges. Avoid double-counting views. Do not return "
        "gram weights. "
        'Return only JSON with keys "dish" and "components"; each component must have "name", '
        '"food_area_fraction", "height_cm_min", "height_cm_max", "shape_factor", "volume_ml_min", '
        '"volume_ml_max", and "confidence".'
    )
    payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(_jpeg_bytes(top)).decode("ascii"),
                       base64.b64encode(_jpeg_bytes(side)).decode("ascii")],
        }],
        "stream": False,
        "think": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 320, "num_ctx": 2048},
        "keep_alive": "10m",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
            response = await client.post(f"{OLLAMA_URL.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
        ollama_result = response.json()
        message = ollama_result.get("message", {})
        answer = message.get("content") or message.get("thinking", "")
        try:
            parsed = parse_component_response(answer)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Qwen returned unusable content=%r thinking=%r", answer[:300], ollama_result.get("message", {}).get("thinking", "")[:200])
            raise ValueError(f"{exc}; response={answer[:160]!r}") from exc
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama. Start it with `ollama serve`.")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        if exc.response.status_code == 404:
            detail = f"Ollama model '{QWEN_MODEL}' is unavailable. Run `ollama pull {QWEN_MODEL}`."
        raise HTTPException(status_code=502, detail=detail)
    except (httpx.TimeoutException, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Qwen response error: {exc}")

    logger.info(
        "vision completed: segmentation=%.2fs qwen=%.2fs total=%.2fs top=%sx%s side=%sx%s",
        segmented_at - started, time.perf_counter() - segmented_at, time.perf_counter() - started,
        top.width, top.height, side.width, side.height,
    )
    estimates = build_weight_estimates(parsed["components"], plate_scale)
    ingredient_totals = {item["name"]: item["weight_g"]["estimate"] for item in estimates}
    return {
        "dish": parsed["dish"],
        "ingredients": ingredient_totals,
        "estimates": estimates,
        "scale_reference": {"plate_diameter_cm": plate_diameter_cm, **plate_scale},
        "method": "Qwen estimates component volumes from two views; backend multiplies volume ranges by food-category density ranges.",
        "limitations": [
            "These are AI-assisted estimates, not measured weights.",
            "Plate calibration depends on a visible, fully framed plate rim; occlusion or perspective can reduce accuracy.",
            "The bundled segmentation checkpoint detects a generic food class, not ingredient-specific masks.",
            "Food density varies with packing, moisture, cooking, and recipe.",
        ],
        "segmentation": {
        "top_view": {"areas": top_areas, "detections": top_detections},
        "side_view": {"areas": side_areas, "detections": side_detections},
        },
    }


def _jpeg_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue()
