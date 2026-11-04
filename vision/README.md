# Food Vision API

FastAPI backend that accepts two photos of one food serving and a plate diameter, runs FoodSeg103 semantic segmentation, asks a local Qwen3-VL model to identify visible ingredients and estimate their geometry, and returns approximate gram weights as JSON.

## Setup on Windows

This guide assumes **Ollama and `qwen3-vl:2b` are already installed**. It also requires Python 3.11, Git, and Git LFS. The segmentation checkpoint is included in this repository using Git LFS.

### 1. Install Git LFS and get this branch

If Git LFS is not installed, install it first from [git-lfs.com](https://git-lfs.com/). Then open PowerShell and run:

```powershell
git lfs install
git clone --branch vision https://github.com/abhirammalyala5-eng/ps.git
cd ps\vision
git lfs pull
```

Check that `models\FoodSeg103-DeepLabV3Plus-MobileNet\best_model.pth` is a real model file (about 18 MB), not a small text pointer.

### 2. Install Python dependencies

Install Python 3.11 if needed, then from the `ps\vision` folder run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch will use CUDA when a compatible NVIDIA setup is available; otherwise the segmentation model runs on CPU and will be slower. The project does not require a separately installed CUDA toolkit.

### 3. Start Ollama and the API

Ollama must be serving `qwen3-vl:2b`. Check it with:

```powershell
ollama list
```

If Ollama is installed but its service is not running, open a second PowerShell window and run `ollama serve`. In the first window, from `ps\vision`, start FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `GET /health` reports whether the segmentation model loaded and whether PyTorch selected CUDA or CPU.

## Try `POST /vision`

Upload:

- `top_image`: a directly overhead photo.
- `side_image`: a view from near plate level.
- `plate_diameter_cm`: the plate's actual diameter (accepted range 15–40 cm).

**Both photos must show the same serving on the same plate.** Keep the entire plate visible, use good lighting, and avoid changing the food between shots. For example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/vision `
  -F "top_image=@C:\path\to\meal-top.jpg" `
  -F "side_image=@C:\path\to\meal-side.jpg" `
  -F "plate_diameter_cm=25"
```

## How it estimates weights

1. A 104-class FoodSeg103 DeepLabV3+ MobileNetV2 model labels image pixels and reports class-area hints for both photos.
2. OpenCV looks for a fully visible plate rim in the top photo. The known plate diameter gives a rough pixel scale and food footprint in square centimeters.
3. Qwen3-VL receives both photos and the segmentation hints. It estimates the dish, visible components, each component's share of the food footprint, and its height as a fraction of plate diameter.
4. The backend converts that height fraction to centimeters, estimates volume from `component footprint × height × shape factor`, then multiplies by a hand-configured food-category density range in `food_density.json` to estimate grams.
5. If plate calibration fails, the API uses Qwen's fallback volume range and marks confidence low. If segmentation labels strongly disagree between views, the response warns that the pair may not show the same serving or that the segmenter may have misclassified food.

The height fraction is capped at 0.20 of the plate diameter as a guard against implausibly tall estimates from the small vision model. This is still a heuristic volume model, not 3D reconstruction or a kitchen-scale measurement.

## Response

- `ingredients`: central estimated grams per component.
- `estimates`: volume range, density range, weight range, confidence, and method per component.
- `scale_reference`: selected plate diameter and detected plate scale, when available.
- `segmentation`: predicted classes, pixel-area shares, and class confidence for each view.
- `limitations`: warnings about calibration, segmentation, density assumptions, or inconsistent views.

The bundled checkpoint has 104 output labels (background plus 103 FoodSeg103 classes). Its [model card](https://huggingface.co/mawiie/food-segmentation-mobilenet) reports 23.33% validation mIoU. It can confuse or miss foods, especially in mixed dishes, so segmentation labels are hints rather than ground truth. The system has not been validated for weight accuracy on a held-out meal dataset; do not claim an accuracy score without weighing test portions.

## Evaluate against a kitchen scale

Create a CSV with columns `ingredient_name,actual_g,predicted_g`, using meals that were not used to tune prompts or thresholds. Then run:

```powershell
.\.venv\Scripts\python.exe evaluate_weights.py weighed_results.csv
```

This reports mean absolute error (MAE) in grams and mean absolute percentage error (MAPE). The separate files in `ML_Model/` and `train/` are not connected to this API; the API does not return calories or macros.

## Main files

- `app.py` — FastAPI routes, segmentation, Ollama request, and weight estimation.
- `models/FoodSeg103-DeepLabV3Plus-MobileNet/` — segmentation checkpoint and class labels.
- `food_density.json` — heuristic density ranges by food category.
- `requirements.txt` — Python packages required to run the API.
- `evaluate_weights.py` — comparison tool for weighed portions.
