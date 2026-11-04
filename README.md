# Food Vision Project

This repository contains the food vision API, its local model files, a separate ML prototype, and the partner-owned frontend.

## Run the API on Windows

1. Make sure Ollama is running and has the Qwen model:

   ```powershell
   ollama pull qwen3-vl:2b
   ```

2. From this project folder, start the backend:

   ```powershell
   .\start_api.ps1
   ```

3. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The API is in `vision/`; that folder is the only source of truth for the backend. `start_api.ps1` launches it using the project environment at `.venv_run`. If the environment has not been installed yet, follow the Windows setup steps in [vision/README.md](vision/README.md).

## Project layout

- `vision/` — FastAPI backend, segmentation weights, and API setup guide.
- `FrontEnd/` — partner-owned frontend. It is not part of backend startup.
- `ML_Model/` and `train/` — separate nutrition prediction experiment; not called by the vision API.
- `models/` — local downloaded model files and caches. The API uses the FoodSeg checkpoint inside `vision/models/`; Ollama manages Qwen separately.

## What the API expects

`POST /vision` accepts two photos of the same serving (`top_image` and `side_image`) and `plate_diameter_cm`. It combines FoodSeg103 semantic segmentation with Ollama Qwen3-VL and returns ingredient and weight estimates as JSON. See [vision/README.md](vision/README.md) for request examples and limitations.
