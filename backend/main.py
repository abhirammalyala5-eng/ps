"""Small FastAPI scaffold for the DietAI image-analysis workflow.

The trained inference artifacts are not available in this repository yet.
This API validates uploads and reports that limitation without inventing results.
"""
import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def configured_origins() -> list[str]:
    value = os.getenv(
        "DIETAI_CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    )
    return [origin.strip() for origin in value.split(",") if origin.strip()]


app = FastAPI(
    title="DietAI India API",
    description="Backend foundation for the DietAI India food image workflow.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Status"])
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""
    return {"status": "ok", "service": "DietAI India API"}


@app.post("/analyze", tags=["Image analysis"])
async def analyze_image(image: UploadFile = File(...)) -> dict[str, str]:
    """Validate an image upload; inference is unavailable until model files arrive."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    contents = await image.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        with Image.open(BytesIO(contents)) as uploaded_image:
            image_format = uploaded_image.format
            uploaded_image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from None

    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Supported image formats are JPEG, PNG, and WEBP.",
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "Image received and validated, but food analysis is not available yet. "
            "The trained model artifacts have not been connected."
        ),
    )
