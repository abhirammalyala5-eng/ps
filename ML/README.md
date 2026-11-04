# NutraVision ML Endpoint

A FastAPI server that predicts nutritional values for food ingredients using pre-trained CatBoost models.

## Pipeline

```
Ingredient text / Vision model JSON
         ↓
  Weight parser  →  TF-IDF (char_wb 2-5) → SVD (64 dims)
         ↓
  5× CatBoost Regressors (one per nutrient)
         ↓
  Scale by weight (predictions are per 100g)
         ↓
  JSON: per-ingredient nutrition + recipe totals
```

## Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Health ping |
| GET | `/health` | Models loaded check |
| POST | `/predict` | Manual ingredient list |
| POST | `/predict/vision` | Vision model JSON — send as-is |
| GET | `/docs` | Swagger UI |

## Setup

### 1. Place model files
Put all model files in the same folder as `app.py`:

```
ML/
├── app.py
├── requirements.txt
├── vectorizer.joblib
├── svd.joblib
├── metadata.joblib
├── attribute_keywords.joblib
├── attribute_names.joblib
├── category_keywords.joblib
├── preparation_keywords.joblib
├── energy_kcal.cbm
├── carb_g.cbm
├── protein_g.cbm
├── fat_g.cbm
└── fibre_g.cbm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Server starts at: `http://localhost:8000`

---

## Usage

### POST `/predict/vision`
Accepts the **raw vision model JSON** without any changes.

```json
{
  "dish": "Biryani",
  "ingredients": {
    "Rice": 440,
    "Chicken": 90,
    "Onion": 30,
    "Cilantro Mint": 20
  }
}
```

**Response:**
```json
{
  "dish": "Biryani",
  "nutrition": [
    { "ingredient": "Rice",    "weight_g": 440, "energy_kcal": 1121.97, "carb_g": 154.98, "protein_g": 25.52, "fat_g": 24.54, "fibre_g": 9.37 },
    { "ingredient": "Chicken", "weight_g": 90,  "energy_kcal": 195.09,  "carb_g": 15.89,  "protein_g": 14.28, "fat_g": 9.57,  "fibre_g": 3.17 }
  ],
  "recipe_totals": {
    "total_weight_g": 580.0,
    "energy_kcal": 1359.75,
    "carb_g": 175.66,
    "protein_g": 42.67,
    "fat_g": 35.29,
    "fibre_g": 12.54
  }
}
```

---

### POST `/predict`
Direct ingredient list input.

```json
{
  "ingredients": [
    { "name": "grilled chicken", "weight_g": 150 },
    { "name": "basmati rice",    "weight_g": 80 },
    { "name": "apple",           "weight_g": 100 }
  ]
}
```

---

## Smart Preparation Detection

The API uses a **3-tier system** to determine how an ingredient is cooked — preventing raw/uncooked assumptions:

| Tier | Method | Example |
|---|---|---|
| 1 | Dish context | Biryani → Rice = boiled |
| 2 | Keyword in name | "grilled chicken" → grilled |
| 3 | Food-type default | "rice" → boiled, "apple" → raw |

Supported dishes: `biryani`, `curry`, `fried rice`, `pulao`, `salad`, `sandwich`, `burger`, `stir fry`, `soup`, `dal`, `pasta`, `pizza`

---

## Predicted Nutrients

| Field | Unit |
|---|---|
| `energy_kcal` | kilocalories |
| `carb_g` | grams |
| `protein_g` | grams |
| `fat_g` | grams |
| `fibre_g` | grams |

> All predictions are made **per 100g** and scaled to the actual ingredient weight.
