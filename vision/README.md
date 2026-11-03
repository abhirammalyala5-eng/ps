# Food Vision API

FastAPI endpoint that accepts a food photo, runs the bundled YOLO segmentation checkpoint, sends the resized photo and pixel-area summary to Ollama's local Qwen3-VL model, and returns JSON ingredient weight estimates.

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

Open `http://127.0.0.1:8000/docs` and upload an image to `POST /vision`.

## Response and limitations

The response contains an `ingredients` object with estimated grams and a `segmentation` object with pixel-area summaries. Weights are visual estimates; a single RGB image cannot reveal actual mass without scale/portion reference data.

The bundled checkpoint metadata currently contains only one class (`food`), despite the upstream model card describing 103 classes. It therefore provides food-region masks, not reliable ingredient classification. Qwen identifies food from the image; this segmentation model currently contributes only a coarse area cue.

For a different Ollama URL/model, set `OLLAMA_URL` and `QWEN_MODEL`. Other optional settings: `FOODSEG_MODEL`, `MAX_IMAGE_BYTES`, `FOODSEG_CONFIDENCE`, `VISION_MAX_SIDE`, and `FOODSEG_IMAGE_SIZE`.
