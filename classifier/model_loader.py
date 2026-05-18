"""
model_loader.py — Hot-swappable ESRB model loader.

Supports two model types:
  v1 — original model: pkl with {model, features, classes}
       predict() takes 31 binary descriptor values only.
  v2 — pipeline model: pkl with {model, has_extra_features: True, ...}
       predict() accepts optional extra features (genre, platform,
       critic_score, year_of_release) for better E/ET discrimination.

HOW TO INSERT A NEW MODEL:
  1. Train with train_model_v1.py or train_model_v2.py
  2. Drop the .pkl into the  models/  directory
  3. Visit /manage-model/ to activate — no restart needed
"""

import os, pickle
import numpy as np
from pathlib import Path
from django.conf import settings

_cache = {}   # filename -> payload


def get_model_dir() -> Path:
    return Path(settings.MODEL_DIR)


def list_models() -> list:
    d = get_model_dir()
    return sorted(f.name for f in d.iterdir() if f.suffix == '.pkl') if d.exists() else []


def load_model(filename: str) -> dict:
    path = get_model_dir() / filename
    mtime = os.path.getmtime(path)
    if filename not in _cache or _cache[filename]['_mtime'] != mtime:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        payload['_mtime'] = mtime
        payload['_filename'] = filename
        _cache[filename] = payload
    return _cache[filename]


def get_active_model_name() -> str | None:
    sentinel = get_model_dir() / '_active_model.txt'
    if sentinel.exists():
        name = sentinel.read_text().strip()
        if name and (get_model_dir() / name).exists():
            return name
    models = list_models()
    return models[0] if models else None


def set_active_model(filename: str):
    (get_model_dir() / '_active_model.txt').write_text(filename)


def get_model_info(filename: str | None = None) -> dict:
    """Return metadata about the active model for display."""
    name = filename or get_active_model_name()
    if not name:
        return {}
    payload = load_model(name)
    return {
        'filename': name,
        'has_extra_features': payload.get('has_extra_features', False),
        'classes': payload.get('classes', []),
        'version': 'v2' if payload.get('has_extra_features') else 'v1',
    }


def predict(binary_values: list, extra: dict | None = None, filename: str | None = None) -> dict:
    """
    Run ESRB classification.

    Parameters
    ----------
    binary_values : list[int]
        Values for each binary descriptor feature (31 values, 0 or 1).
    extra : dict, optional
        Additional features for v2 models:
          genre           — string (e.g. 'Action', 'Sports')
          platform        — string (e.g. 'PC', 'PS4')
          critic_score    — float 0-100 or None
          year_of_release — int or None
          publisher       — string
    filename : str, optional
        Specific .pkl to use; defaults to active model.

    Returns
    -------
    dict with: rating, confidence, all_probs, model_used, confidence_tier
    """
    name = filename or get_active_model_name()
    if not name:
        raise RuntimeError("No model found in models/ directory.")

    payload = load_model(name)
    clf = payload['model']
    classes = payload['classes']

    if payload.get('has_extra_features'):
        X = _build_v2_input(payload, binary_values, extra or {})
    else:
        X = np.array(binary_values, dtype=float).reshape(1, -1)

    proba = clf.predict_proba(X)[0]
    top_idx = int(proba.argmax())
    confidence = round(float(proba[top_idx]) * 100, 1)
    all_probs = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

    # Confidence tier (gap between top-2 matters as much as raw %)
    sorted_p = sorted(proba, reverse=True)
    gap = round((sorted_p[0] - sorted_p[1]) * 100, 1)

    if confidence >= 80 and gap >= 25:
        tier = {'level': 'high', 'label': 'High confidence', 'color': '#10b981'}
    elif confidence >= 60 or gap >= 15:
        tier = {'level': 'moderate', 'label': 'Moderate confidence', 'color': '#f97316'}
    else:
        tier = {'level': 'low', 'label': 'Borderline — review top two', 'color': '#f43f5e'}

    return {
        'rating':          classes[top_idx],
        'confidence':      confidence,
        'all_probs':       all_probs,
        'model_used':      name,
        'confidence_tier': tier,
        'gap':             gap,
    }


# ── private ──────────────────────────────────────────────────────────────────

def _build_v2_input(payload: dict, binary_values: list, extra: dict):
    """Build a DataFrame row for v2 Pipeline models."""
    import pandas as pd

    descriptor_cols   = payload['descriptor_features']
    engineered_cols   = payload.get('engineered_features', [])
    numeric_cols      = payload.get('numeric_features', [])
    categorical_cols  = payload.get('categorical_features', [])
    all_features      = payload['all_features']

    row = dict(zip(descriptor_cols, binary_values))

    # Engineered features computed from binary descriptors
    row['any_blood']     = max(row.get('blood',0), row.get('blood_and_gore',0),
                               row.get('animated_blood',0), row.get('mild_blood',0))
    row['any_violence']  = max(row.get('violence',0), row.get('intense_violence',0),
                               row.get('fantasy_violence',0), row.get('cartoon_violence',0),
                               row.get('mild_violence',0), row.get('mild_fantasy_violence',0),
                               row.get('mild_cartoon_violence',0))
    row['any_sexual']    = max(row.get('sexual_content',0), row.get('sexual_themes',0),
                               row.get('nudity',0), row.get('partial_nudity',0),
                               row.get('suggestive_themes',0), row.get('mild_suggestive_themes',0))
    row['any_language']  = max(row.get('language',0), row.get('strong_janguage',0),
                               row.get('mild_language',0))
    row['any_substance'] = max(row.get('use_of_alcohol',0), row.get('use_of_drugs_and_alcohol',0),
                               row.get('alcohol_reference',0), row.get('drug_reference',0))
    row['mature_score']  = (
        row.get('blood_and_gore',0) * 3 + row.get('intense_violence',0) * 3 +
        row.get('strong_sexual_content',0) * 3 + row.get('nudity',0) * 2 +
        row.get('sexual_content',0) * 2 + row.get('strong_janguage',0) * 2 +
        row.get('blood',0) + row.get('violence',0) + row.get('language',0)
    )
    row['descriptor_count'] = sum(binary_values)

    # Numeric
    cs = extra.get('critic_score')
    yr = extra.get('year_of_release')
    try:
        row['critic_score'] = float(cs) if cs not in (None, '', 'None') else np.nan
    except (ValueError, TypeError):
        row['critic_score'] = np.nan
    try:
        row['year_of_release'] = float(yr) if yr not in (None, '', 'None') else np.nan
    except (ValueError, TypeError):
        row['year_of_release'] = np.nan
    row['has_critic_score'] = 0 if np.isnan(row['critic_score']) else 1

    # Categorical
    row['genre']    = str(extra.get('genre', '')    or 'Unknown').strip() or 'Unknown'
    row['platform'] = str(extra.get('platform', '') or 'Unknown').strip() or 'Unknown'

    return pd.DataFrame([row])[all_features]
