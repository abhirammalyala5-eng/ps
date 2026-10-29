# DietAI India API

This is a beginner-friendly FastAPI foundation for the existing DietAI India frontend. It runs without the colleague's trained model files. The API can check its health and validate JPEG, PNG, and WEBP uploads, but it does **not** make food or nutrition predictions until the real inference code and matching model artifacts are available.

## 1. Create and activate a virtual environment

From the repository root:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the server with the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

## 2. Start the API

With the environment activated:

```powershell
python -m uvicorn main:app --reload
```

The server listens at <http://127.0.0.1:8000>. Interactive API documentation is at <http://127.0.0.1:8000/docs>.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Returns `{"status":"ok","service":"DietAI India API"}` when the API is running. |
| POST | `/analyze` | Accepts one multipart file field named `image`. Validates the upload, then returns HTTP 503 with an honest explanation that inference is not connected. |

The upload limit is 10 MB. Supported formats are JPEG, PNG, and WEBP. Invalid files receive a clear 4xx response. A successful validation currently ends with HTTP 503 because no trained model is available; no predictions are generated.

## Frontend connection

The frontend defaults to `http://127.0.0.1:8000`. It can point to another API host by setting `window.DIETAI_API_URL` before `script.js` loads. For example, add this before the frontend's script tag:

```html
<script>window.DIETAI_API_URL = "http://127.0.0.1:8000";</script>
<script src="script.js"></script>
```

By default, the API allows browser requests from `http://127.0.0.1:5500` and `http://localhost:5500`, common local static-server addresses. Set `DIETAI_CORS_ORIGINS` to a comma-separated list of exact frontend origins when needed. Example in PowerShell:

```powershell
$env:DIETAI_CORS_ORIGINS = "http://127.0.0.1:5500,http://localhost:5500"
```

Keep this setting to the origins you actually use; do not use a wildcard for a deployed site.

## Model integration later

Add the colleague's inference code and model files through the project's agreed secure storage process. Do not commit credentials or private access tokens. Before returning predictions, confirm that the image model, preprocessing, regression model, and matching vectorizer are available and that the output fields match the frontend. This scaffold intentionally contains no fallback or sample predictions.
