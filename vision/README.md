# Food Vision API

FastAPI endpoint that accepts two photos of the same serving (top-down and side-at-plate-level), plus a user-selected plate diameter. It estimates visible plate scale from the top view when the rim is fully in frame, runs a 104-class FoodSeg103 DeepLabV3+ MobileNetV2 semantic segmentation checkpoint on both views, asks Qwen for ingredient names and geometry estimates, and converts those estimates to weight ranges with `food_density.json`.

## Files

- `app.py` — `POST /vision` and `GET /health` implementation.
- `models/FoodSeg103-DeepLabV3Plus-MobileNet/best_model.pth` — multi-class segmentation weights, stored with Git LFS.
- `models/FoodSeg103-DeepLabV3Plus-MobileNet/labels_config.json` — 104-class ID-to-name mapping.
- `food_density.json` — heuristic food-category density ranges in g/ml.
- `evaluate_weights.py` — MAE/MAPE calculator for comparison against kitchen-scale measurements.
- `requirements.txt` — Python dependencies.

## Windows setup

Install Python 3.11, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com/download). Then from this `vision` folder:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
ollama pull qwen3-vl:2b
ollama serve
```

In another PowerShell window:

```powershell
cd path\to\ps\vision
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` and upload both photos to `POST /vision`: `top_image` for a directly overhead photo and `side_image` for a side view at plate level. Both photos must show the same serving. Enter `plate_diameter_cm` as the user's selected plate diameter in centimeters; common choices are 20, 25, and 30 cm (accepted range: 15–40 cm). Keep the entire plate visible in both photos.

Or use curl:

```powershell
curl.exe -X POST http://127.0.0.1:8000/vision `
  -F "top_image=@C:\path\to\meal-top.jpg" `
  -F "side_image=@C:\path\to\meal-side.jpg" `
  -F "plate_diameter_cm=25"
```

## Response and limitations

The response includes:

- `ingredients`: central estimated grams per component.
- `estimates`: estimated volume range, density range, weight range, confidence, and method per component.
- `scale_reference`: selected plate diameter and the top-view circle calibration result, if detected.
- `limitations`: model and measurement caveats.

When the top-view plate rim can be calibrated, the backend estimates each component's volume as `calibrated food footprint × Qwen-estimated component area share × Qwen-estimated side-view height × shape factor`. It multiplies that volume interval by the category's density interval in `food_density.json` to calculate a weight range. If plate calibration fails, it falls back to Qwen's two-view volume interval and marks confidence low. This is an approximate geometric model, not a reconstructed 3D surface; treat results as AI-assisted estimates, not measurements.

Example response shape:

```json
{
  "dish": "chicken biryani",
  "ingredients": {"rice": 220, "chicken": 100},
  "estimates": [{
    "name": "rice",
    "volume_ml": {"min": 220, "estimate": 280, "max": 350},
    "density_g_per_ml": {"min": 0.65, "max": 0.95},
    "weight_g": {"min": 140, "estimate": 220, "max": 330},
    "confidence": "medium",
    "method": "calibrated top-view food footprint × estimated side-view height × shape factor × food-category density range"
  }],
  "scale_reference": {"plate_diameter_cm": 25, "detected": true},
  "limitations": ["These are AI-assisted estimates, not measured weights."]
}
```

The bundled DeepLabV3+ MobileNetV2 checkpoint outputs 104 semantic classes (background plus 103 FoodSeg103 labels). Its [model card](https://huggingface.co/mawiie/food-segmentation-mobilenet) reports 23.33% validation mIoU; expect missed or confused classes, especially for small, mixed, occluded, or visually similar ingredients. The endpoint passes detected ingredient-area summaries to Qwen. This is not a guarantee of ingredient-level accuracy on user photos.

The nutrition training artifacts in the repository are not connected to this endpoint. The API does not return calories or macros; describe that model as a separate module until it is integrated and validated.

## Validate against weighed portions

Capture top and side photos, record the model estimate, then weigh each component with a kitchen scale. Build a CSV with columns `ingredient_name,actual_g,predicted_g` and run:

```powershell
.\.venv\Scripts\python.exe evaluate_weights.py weighed_results.csv
```

The script reports MAE in grams and MAPE for nonzero actual weights. Collect a held-out set of at least 30–50 servings before presenting accuracy metrics; do not report training examples as evaluation.

For a different Ollama URL/model, set `OLLAMA_URL` and `QWEN_MODEL`. The API requests a 6144-token context by default; change it with `QWEN_NUM_CTX` if Ollama reports that the prompt exceeds its context window. Other optional settings: `FOODSEG_MODEL`, `MAX_IMAGE_BYTES`, `FOODSEG_CONFIDENCE`, `VISION_MAX_SIDE`, and `FOODSEG_IMAGE_SIZE`.
