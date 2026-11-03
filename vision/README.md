# Food Vision API

FastAPI endpoint that accepts two photos of the same serving (top-down and side-at-plate-level), runs the bundled YOLO segmentation checkpoint on both views, sends both resized photos and their pixel-area summaries to Ollama's local Qwen3-VL model, and returns JSON ingredient weight estimates. The two views let Qwen use top-view footprint and side-view height to make a volume-informed estimate.

## Files

- `app.py` — `POST /vision` and `GET /health` implementation.
- `models/FoodSeg103-YOLO11l/best.pt` — segmentation weights, stored with Git LFS.
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
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
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

The response contains an `ingredients` object with estimated grams, the supplied `scale_reference`, and separate top/side segmentation summaries. Two views and a known plate diameter make volume estimation more grounded, but mass is still inferred using likely food density; exact weight requires a scale.

The bundled checkpoint metadata currently contains only one class (`food`), despite the upstream model card describing 103 classes. It therefore provides food-region masks, not reliable ingredient classification. Qwen identifies food from the image; this segmentation model currently contributes only a coarse area cue.

For a different Ollama URL/model, set `OLLAMA_URL` and `QWEN_MODEL`. Other optional settings: `FOODSEG_MODEL`, `MAX_IMAGE_BYTES`, `FOODSEG_CONFIDENCE`, `VISION_MAX_SIDE`, and `FOODSEG_IMAGE_SIZE`.
