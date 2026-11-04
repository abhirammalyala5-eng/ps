# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
🥦 NutraVision Prediction API
==============================
FastAPI endpoint that accepts a JSON list of ingredients and returns
per-ingredient nutritional info + recipe totals.

Pipeline:
  1. Parse ingredient text  → weight_g, food_name, preparation, category, attributes
  2. TF-IDF (char_wb 2-5)  → TruncatedSVD (64 dims)
  3. CatBoostRegressor      → 5 targets (energy_kcal, carb_g, protein_g, fat_g, fibre_g)
  4. Scale by weight (predictions are per 100 g)
"""

import re
import os
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")  # suppress sklearn version warnings

# ─────────────────────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent          # same folder as app.py

TARGETS = ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g"]

# ─────────────────────────────────────────────────────────────
# 2.  LOAD ARTIFACTS  (once at startup)
# ─────────────────────────────────────────────────────────────
print("Loading NutraVision model artifacts...")

vectorizer          = joblib.load(MODEL_DIR / "vectorizer.joblib")
svd                 = joblib.load(MODEL_DIR / "svd.joblib")
attribute_keywords  = joblib.load(MODEL_DIR / "attribute_keywords.joblib")
attribute_names     = joblib.load(MODEL_DIR / "attribute_names.joblib")
category_keywords   = joblib.load(MODEL_DIR / "category_keywords.joblib")
preparation_keywords= joblib.load(MODEL_DIR / "preparation_keywords.joblib")
metadata            = joblib.load(MODEL_DIR / "metadata.joblib")

catboost_models: Dict[str, CatBoostRegressor] = {}
for target in TARGETS:
    cb = CatBoostRegressor()
    cb.load_model(str(MODEL_DIR / f"{target}.cbm"))
    catboost_models[target] = cb

print(f"[OK] Loaded {len(catboost_models)} CatBoost models | SVD dims: {svd.n_components}")

# ─────────────────────────────────────────────────────────────
# 3.  PARSING HELPERS
# ─────────────────────────────────────────────────────────────
_WEIGHT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:g|grams?|kg|kilograms?|ml|millilitres?|oz|ounces?|lb|lbs?|pounds?)",
    re.IGNORECASE,
)
_G_PER_UNIT = {
    "g": 1, "gram": 1, "grams": 1,
    "kg": 1000, "kilogram": 1000, "kilograms": 1000,
    "ml": 1,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "lbs": 453.59, "pound": 453.59, "pounds": 453.59,
}


def _parse_weight(text: str) -> tuple[float, str]:
    """Return (weight_g, text_without_weight_token)."""
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|ml|millilitres?|oz|ounces?|lb|lbs?|pounds?)\b",
        text, re.IGNORECASE,
    )
    if m:
        qty = float(m.group(1))
        unit = m.group(2).lower().rstrip("s") if m.group(2).lower() not in _G_PER_UNIT else m.group(2).lower()
        # normalise unit
        for key in _G_PER_UNIT:
            if m.group(2).lower().startswith(key[:3]):
                weight_g = qty * _G_PER_UNIT[key]
                break
        else:
            weight_g = qty
        cleaned = (text[: m.start()] + text[m.end():]).strip(" ,")
        return weight_g, cleaned
    return 100.0, text          # default 100 g


def _detect_preparation(text: str) -> str:
    text_l = text.lower()
    for prep, keywords in preparation_keywords.items():
        for kw in keywords:
            if kw in text_l:
                return prep
    return "unknown"


def _detect_category(text: str) -> str:
    text_l = text.lower()
    for cat in category_keywords:
        if cat in text_l:
            return cat
    return "unknown"


def _detect_attributes(text: str) -> List[int]:
    text_l = text.lower()
    flags = []
    for attr in attribute_names:
        kws = attribute_keywords.get(attr, [])
        flags.append(1 if any(kw in text_l for kw in kws) else 0)
    return flags


ATTR_COLS = [f"attr_{a}" for a in attribute_names]
CAT_FEATURES = ["category", "preparation"]

# ── Dish → default preparation context ───────────────────────
# Maps dish keywords → { ingredient keyword → preparation }
DISH_PREP_CONTEXT: Dict[str, Dict[str, str]] = {
    "biryani":      {"rice": "boiled", "chicken": "roasted", "mutton": "roasted",
                     "lamb": "roasted", "egg": "boiled", "onion": "fried",
                     "potato": "boiled"},
    "curry":        {"chicken": "boiled", "mutton": "boiled", "lamb": "boiled",
                     "paneer": "boiled", "potato": "boiled", "rice": "boiled"},
    "fried rice":   {"rice": "boiled", "egg": "fried", "chicken": "fried",
                     "vegetable": "fried", "onion": "fried"},
    "pulao":        {"rice": "boiled", "chicken": "boiled", "vegetable": "boiled"},
    "salad":        {"chicken": "grilled", "egg": "boiled", "tuna": "raw",
                     "vegetable": "raw", "lettuce": "raw", "tomato": "raw"},
    "sandwich":     {"chicken": "grilled", "egg": "boiled", "vegetable": "raw"},
    "burger":       {"chicken": "fried", "beef": "fried", "patty": "fried"},
    "stir fry":     {"chicken": "fried", "beef": "fried", "vegetable": "fried",
                     "noodle": "boiled", "rice": "boiled"},
    "soup":         {"chicken": "boiled", "vegetable": "boiled", "potato": "boiled"},
    "dal":          {"lentil": "boiled", "dal": "boiled", "onion": "fried"},
    "pasta":        {"pasta": "boiled", "chicken": "grilled", "vegetable": "boiled"},
    "pizza":        {"chicken": "roasted", "cheese": "raw", "vegetable": "roasted"},
}


def _get_preparation_for_dish(dish: str, ingredient: str) -> Optional[str]:
    """Return the most appropriate preparation for an ingredient given the dish."""
    dish_l = dish.lower()
    ingr_l = ingredient.lower()
    for dish_key, ingr_map in DISH_PREP_CONTEXT.items():
        if dish_key in dish_l:
            for ingr_key, prep in ingr_map.items():
                if ingr_key in ingr_l:
                    return prep
    return None


# ── Tier 3: Ingredient-type smart defaults ────────────────────
# When no dish context and no prep keyword in name, use this.
INGREDIENT_DEFAULT_PREP: Dict[str, str] = {
    # Grains (always served cooked)
    "rice": "boiled",    "basmati": "boiled", "noodle": "boiled",
    "pasta": "boiled",   "quinoa": "boiled",  "oat": "boiled",
    "bread": "baked",    "roti": "baked",     "naan": "baked",
    "chapati": "baked",  "paratha": "fried",
    # Proteins (almost always cooked)
    "chicken": "boiled", "mutton": "boiled",  "lamb": "boiled",
    "beef": "boiled",    "pork": "boiled",    "fish": "boiled",
    "prawn": "boiled",   "shrimp": "boiled",  "egg": "boiled",
    "tuna": "raw",       "salmon": "raw",
    # Legumes (always cooked)
    "dal": "boiled",     "lentil": "boiled",  "bean": "boiled",
    "chickpea": "boiled","pea": "boiled",
    # Vegetables (default boiled)
    "potato": "boiled",  "carrot": "boiled",  "spinach": "boiled",
    "onion": "boiled",   "tomato": "boiled",  "broccoli": "boiled",
    "cauliflower": "boiled", "cabbage": "boiled", "corn": "boiled",
    # Fruits (raw)
    "apple": "raw",   "banana": "raw",  "mango": "raw",
    "orange": "raw",  "grape": "raw",   "berry": "raw",
    # Dairy (raw)
    "milk": "raw",    "cheese": "raw",  "paneer": "raw",
    "yogurt": "raw",  "curd": "raw",    "butter": "raw",
}


def _ingredient_default_prep(food_name: str) -> Optional[str]:
    """Return smart default preparation based on ingredient type."""
    name_l = food_name.lower()
    for keyword, prep in INGREDIENT_DEFAULT_PREP.items():
        if keyword in name_l:
            return prep
    return None


def _resolve_preparation(dish: Optional[str], food_name: str) -> str:
    """
    3-tier preparation resolution — prevents 'unknown' for common foods:
      1. Dish context      e.g. Biryani  → rice = boiled
      2. Name keyword      e.g. 'grilled chicken' → grilled
      3. Ingredient default e.g. 'chicken' → boiled
      4. Final fallback    'boiled'
    """
    if dish:
        prep = _get_preparation_for_dish(dish, food_name)
        if prep:
            return prep
    prep = _detect_preparation(food_name)
    if prep != "unknown":
        return prep
    prep = _ingredient_default_prep(food_name)
    if prep:
        return prep
    return "boiled"


def _build_pool(food_name: str, category: str, preparation: str, attributes: List[int]) -> Pool:
    """Build a CatBoost Pool matching the model's expected feature schema."""
    tfidf_vec = vectorizer.transform([food_name])
    svd_vec   = svd.transform(tfidf_vec)[0]        # shape (64,)
    text_cols = {f"text_{i}": float(svd_vec[i]) for i in range(len(svd_vec))}
    cat_col   = {"category": str(category), "preparation": str(preparation)}
    attr_col  = {col: int(val) for col, val in zip(ATTR_COLS, attributes)}
    df = pd.DataFrame([{**text_cols, **cat_col, **attr_col}])
    return Pool(df, cat_features=CAT_FEATURES)


# ─────────────────────────────────────────────────────────────
# 4.  FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="NutraVision Prediction API",
    description=(
        "Predict nutritional values for a list of ingredients using "
        "CatBoost regressors trained on a food nutrition dataset.\n\n"
        "Predictions are **per 100 g** then scaled to the supplied weight.\n\n"
        "**Supported targets:** energy_kcal, carb_g, protein_g, fat_g, fibre_g"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────
class IngredientIn(BaseModel):
    name: str = Field(
        ...,
        description='Free-text ingredient, e.g. "150g grilled chicken" or just "apple"',
        examples=["150g grilled chicken", "50g basmati rice", "apple"],
    )
    weight_g: Optional[float] = Field(
        None,
        description="Explicit weight override in grams (overrides any amount parsed from name)",
        examples=[100, 250],
    )
    category: Optional[str] = Field(
        None,
        description="Food category hint (poultry/meat/seafood/grain/vegetable/fruit/dairy/egg/legume/nut)",
    )
    preparation: Optional[str] = Field(
        None,
        description="Preparation hint (raw/boiled/grilled/roasted/fried/steamed/baked/dried/unknown)",
    )


class NutrientResult(BaseModel):
    entered_item:  str
    parsed_food:   str
    weight_g:      float
    category:      str
    preparation:   str
    energy_kcal:   float
    carb_g:        float
    protein_g:     float
    fat_g:         float
    fibre_g:       float


class RecipeTotals(BaseModel):
    total_weight_g: float
    energy_kcal:    float
    carb_g:         float
    protein_g:      float
    fat_g:          float
    fibre_g:        float


class PredictRequest(BaseModel):
    ingredients: List[IngredientIn] = Field(
        ...,
        description="List of ingredients to analyse",
        min_length=1,
    )


class PredictResponse(BaseModel):
    ingredients:   List[NutrientResult]
    recipe_totals: RecipeTotals


# ── Routes ───────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="Root ping")
def root():
    return {
        "service": "NutraVision Prediction API",
        "status":  "running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"], summary="Health check")
def health():
    return {
        "status":  "ok",
        "models_loaded": list(catboost_models.keys()),
        "svd_components": svd.n_components,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Nutrition"],
    summary="Predict nutritional values for a list of ingredients",
    description=(
        "Accepts a JSON list of ingredients and returns:\n"
        "- **Per-ingredient breakdown**: energy_kcal, carb_g, protein_g, fat_g, fibre_g\n"
        "- **Recipe totals**: summed values across all ingredients\n\n"
        "| Input style | Behaviour |\n"
        "|---|---|\n"
        "| `\"150g grilled chicken\"` | 150 g, food = grilled chicken |\n"
        "| `\"50 grams basmati rice\"` | 50 g, food = basmati rice |\n"
        "| `\"apple\"` | defaults to **100 g** |\n"
        "| `{\"name\": \"broccoli\", \"weight_g\": 80}` | explicit 80 g override |"
    ),
)
def predict(request: PredictRequest) -> PredictResponse:
    results: List[NutrientResult] = []
    totals = {t: 0.0 for t in TARGETS}
    total_weight = 0.0

    for item in request.ingredients:
        # 4a. Parse weight + clean food name
        weight_g, food_name = _parse_weight(item.name)
        if item.weight_g is not None:
            weight_g = item.weight_g         # explicit override wins

        # 4b. Detect / override preparation & category
        preparation = item.preparation or _resolve_preparation(None, food_name)
        category    = item.category    or _detect_category(item.name)
        attributes  = _detect_attributes(item.name)

        # 4c. Build CatBoost Pool
        try:
            pool = _build_pool(food_name, category, preparation, attributes)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Feature extraction failed for '{item.name}': {exc}")

        # 4d. Run CatBoost models  (predictions are per 100 g)
        preds_per100g: Dict[str, float] = {}
        for target, model in catboost_models.items():
            try:
                val = float(model.predict(pool)[0])
                preds_per100g[target] = max(val, 0.0)   # clamp negatives
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Model error for '{target}': {exc}")

        # 4e. Scale to actual weight
        scale = weight_g / 100.0
        scaled = {t: round(v * scale, 2) for t, v in preds_per100g.items()}

        results.append(NutrientResult(
            entered_item=item.name,
            parsed_food=food_name,
            weight_g=weight_g,
            category=category,
            preparation=preparation,
            **scaled,
        ))

        total_weight += weight_g
        for t in TARGETS:
            totals[t] += scaled[t]

    return PredictResponse(
        ingredients=results,
        recipe_totals=RecipeTotals(
            total_weight_g=round(total_weight, 2),
            **{t: round(v, 2) for t, v in totals.items()},
        ),
    )


# ─────────────────────────────────────────────────────────────
# 5.  VISION MODEL ENDPOINT
# Accepts the raw JSON from the vision model unchanged.
# Reads ingredients from:
#   - "ingredients" dict  → { "Rice": 440, "Chicken": 90, ... }
#   - falls back to "estimates" array → weight_g.estimate per item
# ─────────────────────────────────────────────────────────────

class VisionRequest(BaseModel):
    """Accepts the full vision model JSON as-is. Only 'ingredients' is required."""
    dish:        Optional[str]       = None
    ingredients: Optional[Dict[str, Any]] = None   # { "Rice": 440, ... }
    estimates:   Optional[List[Dict[str, Any]]] = None
    # all other vision fields (scale_reference, segmentation, etc.) are ignored
    model_config = {"extra": "allow"}              # allow unknown fields


class VisionNutrientResult(BaseModel):
    ingredient:  str
    weight_g:    float
    energy_kcal: float
    carb_g:      float
    protein_g:   float
    fat_g:       float
    fibre_g:     float


class VisionPredictResponse(BaseModel):
    dish:          Optional[str]
    nutrition:     List[VisionNutrientResult]
    recipe_totals: RecipeTotals


@app.post(
    "/predict/vision",
    response_model=VisionPredictResponse,
    tags=["Nutrition"],
    summary="Predict nutrition from vision model JSON",
    description=(
        "Accepts the **raw JSON output from the vision model** unchanged.\n\n"
        "Extracts ingredient names + weights from the `ingredients` dict "
        "`{ 'Rice': 440, 'Chicken': 90 }` and returns nutritional predictions.\n\n"
        "Falls back to `estimates[].weight_g.estimate` if `ingredients` is missing."
    ),
)
def predict_vision(request: VisionRequest) -> VisionPredictResponse:
    # ── 1. Extract ingredient → weight pairs ──────────────────
    pairs: List[tuple[str, float]] = []

    if request.ingredients:
        # Primary source: { "Rice": 440, "Chicken": 90, ... }
        for name, weight in request.ingredients.items():
            try:
                pairs.append((str(name), float(weight)))
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid weight for ingredient '{name}': {weight}"
                )
    elif request.estimates:
        # Fallback: estimates[].weight_g.estimate
        for est in request.estimates:
            name = est.get("name", "unknown")
            wg = est.get("weight_g", {})
            if isinstance(wg, dict):
                weight = wg.get("estimate", 100.0)
            else:
                weight = float(wg) if wg else 100.0
            pairs.append((str(name), float(weight)))
    else:
        raise HTTPException(
            status_code=422,
            detail="Request must contain either 'ingredients' dict or 'estimates' array."
        )

    # ── 2. Run predictions (same pipeline as /predict) ────────
    results: List[VisionNutrientResult] = []
    totals = {t: 0.0 for t in TARGETS}
    total_weight = 0.0

    for food_name, weight_g in pairs:
        # Full 3-tier preparation resolution
        preparation = _resolve_preparation(request.dish, food_name)
        category   = _detect_category(food_name)
        attributes = _detect_attributes(food_name)

        try:
            pool = _build_pool(food_name, category, preparation, attributes)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Feature error for '{food_name}': {exc}")

        preds_per100g: Dict[str, float] = {}
        for target, model in catboost_models.items():
            try:
                val = float(model.predict(pool)[0])
                preds_per100g[target] = max(val, 0.0)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Model error for '{target}': {exc}")

        scale  = weight_g / 100.0
        scaled = {t: round(v * scale, 2) for t, v in preds_per100g.items()}

        results.append(VisionNutrientResult(
            ingredient=food_name,
            weight_g=weight_g,
            **scaled,
        ))

        total_weight += weight_g
        for t in TARGETS:
            totals[t] += scaled[t]

    return VisionPredictResponse(
        dish=request.dish,
        nutrition=results,
        recipe_totals=RecipeTotals(
            total_weight_g=round(total_weight, 2),
            **{t: round(v, 2) for t, v in totals.items()},
        ),
    )


# ─────────────────────────────────────────────────────────────
# 6.  DEV ENTRY-POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

