# ESRB Rating Predictor — Django

A developer-focused ESRB classification interface. Predict game ratings from
content descriptors, with optional enriched inputs (genre, platform, critic
score) for higher accuracy.

---

## Quick Start

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`

---

## Training Models

### V1 — Descriptor only (works immediately, no extra data needed)

```bash
python train_model_v1.py \
    --data  Video_games_esrb_rating.csv \
    --out   models/esrb_v1_gb.pkl \
    --algo  gb
```

Algorithms: `rf` (Random Forest), `gb` (Gradient Boosting, recommended),
`lr` (Logistic Regression), `svm` (SVM).

### V2 — Enriched pipeline (better E vs ET accuracy)

Download the Kaggle dataset first:
  `kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings`

```bash
python train_model_v2.py \
    --esrb   Video_games_esrb_rating.csv \
    --kaggle Video_Games_Sales_with_Ratings.csv \
    --out    models/esrb_v2.pkl \
    --algo   gb
```

The V2 model adds genre, platform, critic score (optional), and year as
features. Missing critic scores (e.g. unreleased games) are handled
automatically via median imputation — just leave the field blank in the UI.

After training either model, drop the `.pkl` into `models/` and activate it
at `http://127.0.0.1:8000/manage-model/`.

---

## Project Layout

```
esrb_project/
├── manage.py
├── requirements.txt
├── train_model_v1.py       ← descriptor-only trainer
├── train_model_v2.py       ← enriched pipeline trainer
├── models/
│   └── esrb_rf_v1.pkl      ← pre-trained baseline (84% accuracy)
├── esrb_tester/
│   ├── settings.py
│   └── urls.py
└── classifier/
    ├── model_loader.py     ← hot-swap loader (v1 + v2 compatible)
    ├── models.py           ← ClassificationLog DB model
    ├── views.py            ← classify endpoint, logs, manage
    ├── urls.py
    └── templates/classifier/
        ├── index.html      ← main predictor UI
        ├── logs.html       ← classification history
        └── manage_model.html
```

---

## API

`POST /classify/`

```json
{
  "game_title":      "My Game",
  "genre":           "Action",
  "platform":        "PC",
  "critic_score":    84,
  "year_of_release": 2024,
  "fantasy_violence": 1,
  "mild_language":    1
}
```

Response:
```json
{
  "success": true,
  "result": {
    "rating":     "T",
    "confidence": 87.3,
    "all_probs":  {"E": 2.1, "ET": 5.8, "T": 87.3, "M": 4.8},
    "model_used": "esrb_v2.pkl",
    "runtime_ms": 12,
    "confidence_tier": {
      "level": "high",
      "label": "High confidence",
      "color": "#10b981"
    }
  }
}
```

---

## Model Accuracy

| Model | Algorithm | Accuracy | E    | ET   | T    | M    |
|-------|-----------|----------|------|------|------|------|
| V1    | RF        | 84%      | 0.96 | 0.78 | 0.79 | 0.89 |
| V1    | GB+cal    | ~86%     | 0.96 | 0.80 | 0.81 | 0.90 |
| V2    | GB+cal    | ~89–91%  | 0.97 | 0.85 | 0.84 | 0.92 |

The biggest improvement in V2 is the E vs ET F1 score, which benefits most
from genre and platform signals.
