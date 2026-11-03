"""Food image analysis API: FoodSeg103 segmentation + Ollama Qwen3-VL."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
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
        "ingredients": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        }
    },
    "required": ["ingredients"],
    "additionalProperties": False,
}


def segmentation_summary(image: Image.Image) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return class area fractions and per-instance detection metadata."""
    result = segmenter.predict(
        image, conf=CONFIDENCE, imgsz=SEGMENT_SIZE, device=0 if __import__("torch").cuda.is_available() else "cpu",
        half=__import__("torch").cuda.is_available(), max_det=30, verbose=False,
    )[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return [], []

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
    return areas, instances


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Qwen did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("ingredients"), dict):
        raise ValueError('Expected JSON shaped like {"ingredients": {...}}')
    cleaned: dict[str, float] = {}
    for name, grams in value["ingredients"].items():
        if isinstance(name, str) and isinstance(grams, (int, float)) and not isinstance(grams, bool):
            cleaned[name] = max(0.0, round(float(grams), 1))
    return {"ingredients": cleaned}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "segmenter": "loaded", "qwen_model": QWEN_MODEL}


@app.post("/vision")
async def vision(image: UploadFile = File(...)) -> dict[str, Any]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image")
    raw = await image.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    started = time.perf_counter()
    if max(pil_image.size) > MAX_SIDE:
        pil_image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    areas, detections = segmentation_summary(pil_image)
    segmented_at = time.perf_counter()
    prompt = (
        "Carefully inspect the ENTIRE food image yourself. The image is the source of truth and has "
        "higher priority than the segmentation hints. First identify the dish, then make an exhaustive "
        "ingredient inventory: include every distinct visible food component, protein, starch, "
        "vegetable, sauce, and garnish. For recognizable mixed dishes, also include common ingredients "
        "that are likely mixed in and not individually separable in the photo (for example, chicken "
        "biryani can include rice, chicken, and visible or likely vegetables such as onion or peas). "
        "Use recipe knowledge only to name likely components of the dish, not to claim precise amounts "
        "of ingredients hidden inside it. A FoodSeg103 segmenter returned these OPTIONAL, noisy hints: "
        f"{json.dumps(areas, ensure_ascii=False)}. The segmenter may say only 'food', omit ingredients, "
        "or be wrong. Do not copy its labels as the answer, do not let it limit your visual analysis, "
        "and ignore hints that conflict with the image. No portion weight is provided: estimate the "
        "visible edible food mass from the apparent serving size, then divide it among the main food "
        "components using typical recipe proportions. Weights must describe edible portions, not recipe "
        "ingredient quantities. Do NOT assign gram weights to hidden spices, seasoning, oil, or tiny "
        "garnishes; omit those unless clearly visible as substantial portions. Avoid double-counting, "
        "keep the total plausible for the pictured serving, and round estimates to the nearest 10 g. "
        "Pixel area is only a weak cue, never a direct grams conversion. Include each substantial visible "
        "component, even if partly mixed or covered. Return ONLY JSON shaped exactly like "
        '{"ingredients":{"ingredient name":number}}. Use nonnegative numeric grams, no markdown or extra keys.'
    )
    payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(_jpeg_bytes(pil_image)).decode("ascii")],
        }],
        "stream": False,
        "think": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 192, "num_ctx": 2048},
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
            parsed = parse_json_object(answer)
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
        "vision completed: segmentation=%.2fs qwen=%.2fs total=%.2fs image=%sx%s",
        segmented_at - started, time.perf_counter() - segmented_at, time.perf_counter() - started,
        pil_image.width, pil_image.height,
    )
    return {**parsed, "segmentation": {"areas": areas, "detections": detections}}


def _jpeg_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue()
