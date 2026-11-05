# Food Vision API

FastAPI backend that accepts two photos of the same food serving and the plate diameter, runs FoodSeg103 semantic segmentation, asks a local Qwen3-VL model to identify visible ingredients and estimate their geometry, and returns approximate gram weights as JSON.

## Windows setup

This guide assumes **Ollama and `qwen3-vl:2b` are already installed**. A new computer also needs Python 3.11 (64-bit), Git for Windows, and Git LFS:

- [Python 3.11](https://www.python.org/downloads/windows/)
- [Git for Windows](https://git-scm.com/download/win)
- [Git LFS](https://git-lfs.com/) to download the segmentation checkpoint

### 1. Clone the `vision` branch

Open PowerShell and run:

```powershell
git lfs install
git clone --branch vision https://github.com/abhirammalyala5-eng/ps.git
cd ps\vision
git lfs pull
```

Confirm `models\FoodSeg103\best_model.pth` is about 18 MB. If it is a small text file, Git LFS did not download the checkpoint.

### 2. Install the API's Python packages

From the `ps\vision` folder:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The segmentation model uses CUDA if the installed PyTorch build and NVIDIA driver support it; otherwise it runs on CPU and takes longer. No separate CUDA toolkit is required.

### 3. Start Ollama and the API

Check that the Qwen model is installed:

```powershell
ollama list
```

If the Ollama service is not already running, open a second PowerShell window and run `ollama serve`. In the first window, from `ps\vision`, start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `GET /health` reports the segmentation device, tile settings, image pixel limit, and Qwen context size.

## Use `POST /vision`

Provide:

- `top_image`: a photo taken directly above the food.
- `side_image`: a view of the food from near plate level.
- `plate_diameter_cm`: the actual plate diameter, between 15 and 40 cm.

**Both images must show the same serving on the same plate.** Keep the plate fully visible in both photos.

```powershell
curl.exe -X POST http://127.0.0.1:8000/vision `
  -F "top_image=@C:\path\to\meal-top.jpg" `
  -F "side_image=@C:\path\to\meal-side.jpg" `
  -F "plate_diameter_cm=25"
```

## Image handling and context

- The API no longer downsizes or crops uploads. It forwards the original uploaded bytes to Ollama without JPEG recompression.
- FoodSeg processes the original-resolution image in 512 × 512 overlapping tiles (64-pixel overlap). This keeps the original pixel scale, covers the entire frame, and avoids stretching a landscape photo into a square. Tiling can make segmentation slower, especially for large photos.
- Uploads larger than 40 MB or 50 megapixels are rejected with an error; they are not silently resized. These limits can be raised with `MAX_IMAGE_BYTES` and `MAX_IMAGE_PIXELS` if the computer has enough memory.
- Qwen's Ollama request uses a 16,384-token context by default (`QWEN_NUM_CTX`). Raising it may help with very large photos, but increases memory use. Qwen/Ollama still performs its own internal vision encoding; the API cannot control how the model represents image pixels internally.

## How weights are estimated

1. FoodSeg103 labels pixels and reports class-area hints for both views.
2. OpenCV searches the top image for the plate rim. The known diameter provides a rough physical scale and food footprint if the rim is visible.
3. Qwen receives both original photos and the segmentation hints. It estimates the dish, visible components, each component's footprint share, and height as a fraction of the plate diameter.
4. The backend converts the height fraction to centimeters, estimates volume from `component footprint × height × shape factor`, then multiplies by a hand-configured food-category density range in `food_density.json` to estimate grams.
5. If plate calibration fails, the endpoint falls back to Qwen's estimated volume range and lowers confidence. If FoodSeg labels conflict between the views, the response warns the caller to check the photo pair.

This is a heuristic volume estimate, not a 3D reconstruction or a kitchen-scale measurement. The density ranges are category-level assumptions, not measured values for every ingredient.

## Response

- `ingredients`: central estimated grams per component.
- `estimates`: volume and weight ranges, density assumptions, confidence, and calculation method.
- `scale_reference`: detected plate scale, if available.
- `segmentation`: classes, pixel-area percentages, and confidence for each view.
- `image_processing`: original input dimensions and confirmation of the image path used by each model.
- `limitations`: warnings about image mismatch, calibration, and model uncertainty.

The bundled FoodSeg103 checkpoint has 104 output labels (background plus 103 food labels). Its [model card](https://huggingface.co/mawiie/food-segmentation-mobilenet) reports 23.33% validation mIoU. It can confuse or miss foods, particularly in mixed meals. That metric is not the accuracy of the final weight estimates. Do not claim a weight-accuracy score without comparing predictions with kitchen-scale measurements on a held-out dataset.

## Evaluate against a kitchen scale

Create a CSV with columns `ingredient_name,actual_g,predicted_g`, then run:

```powershell
.\.venv\Scripts\python.exe evaluate_weights.py weighed_results.csv
```

This reports MAE in grams and MAPE for nonzero actual weights. The separate training files in `ML_Model/` and `train/` are not connected to this API; it does not return calories or macros.

## Configuration

Defaults can be changed with environment variables: `OLLAMA_URL`, `QWEN_MODEL`, `QWEN_NUM_CTX`, `MAX_IMAGE_BYTES`, `MAX_IMAGE_PIXELS`, `FOODSEG_MODEL`, `FOODSEG_LABELS`, `FOODSEG_CONFIDENCE`, `FOODSEG_TILE_SIZE`, and `FOODSEG_TILE_OVERLAP`.
