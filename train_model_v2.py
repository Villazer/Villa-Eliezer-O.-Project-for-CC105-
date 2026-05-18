"""
train_model_v2.py — Enriched ESRB classifier using merged datasets.

Combines:
  - Video_games_esrb_rating.csv  (descriptors + rating)
  - Video_Games_Sales_with_Ratings.csv  (genre, platform, critic score)

The merge is done on normalized title. Games that don't match the Kaggle
dataset are kept but their enriched fields (genre, critic_score, etc.) will
be NaN — the sklearn Pipeline handles this automatically via imputation.

This directly fixes the E vs ET ambiguity by adding genre + platform signals.

Usage
-----
  python train_model_v2.py \
      --esrb   Video_games_esrb_rating.csv \
      --kaggle Video_Games_Sales_with_Ratings.csv \
      --out    models/esrb_v2.pkl \
      --algo   gb

Download the Kaggle dataset from:
  kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
"""

import argparse, pickle, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

# ── Feature definitions ───────────────────────────────────────────────────────

DESCRIPTOR_COLS = [
    'alcohol_reference','animated_blood','blood','blood_and_gore',
    'cartoon_violence','crude_humor','drug_reference','fantasy_violence',
    'intense_violence','language','lyrics','mature_humor','mild_blood',
    'mild_cartoon_violence','mild_fantasy_violence','mild_language',
    'mild_lyrics','mild_suggestive_themes','mild_violence','no_descriptors',
    'nudity','partial_nudity','sexual_content','sexual_themes',
    'simulated_gambling','strong_janguage','strong_sexual_content',
    'suggestive_themes','use_of_alcohol','use_of_drugs_and_alcohol','violence',
]

ENGINEERED_COLS = [
    'any_blood','any_violence','any_sexual','any_language',
    'any_substance','mature_score','descriptor_count','has_critic_score',
]

NUMERIC_COLS      = ['critic_score','year_of_release']
CATEGORICAL_COLS  = ['genre','platform']

ALL_FEATURES = DESCRIPTOR_COLS + ENGINEERED_COLS + NUMERIC_COLS + CATEGORICAL_COLS

ALGOS = {
    'rf':  lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'gb':  lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, random_state=42),
    'lr':  lambda: LogisticRegression(max_iter=2000, class_weight='balanced'),
    'svm': lambda: SVC(probability=True, class_weight='balanced'),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_title(t: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = str(t).lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['any_blood']     = df[['blood','blood_and_gore','animated_blood','mild_blood']].max(axis=1)
    df['any_violence']  = df[['violence','intense_violence','fantasy_violence',
                               'cartoon_violence','mild_violence',
                               'mild_fantasy_violence','mild_cartoon_violence']].max(axis=1)
    df['any_sexual']    = df[['sexual_content','sexual_themes','nudity',
                               'partial_nudity','suggestive_themes',
                               'mild_suggestive_themes']].max(axis=1)
    df['any_language']  = df[['language','strong_janguage','mild_language']].max(axis=1)
    df['any_substance'] = df[['use_of_alcohol','use_of_drugs_and_alcohol',
                               'alcohol_reference','drug_reference']].max(axis=1)
    df['mature_score']  = (
        df['blood_and_gore']        * 3 +
        df['intense_violence']      * 3 +
        df['strong_sexual_content'] * 3 +
        df['nudity']                * 2 +
        df['sexual_content']        * 2 +
        df['strong_janguage']       * 2 +
        df['blood']                 * 1 +
        df['violence']              * 1 +
        df['language']              * 1
    )
    df['descriptor_count'] = df[DESCRIPTOR_COLS].sum(axis=1)
    df['has_critic_score'] = df['critic_score'].notna().astype(int)
    return df


def load_esrb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in DESCRIPTOR_COLS if c not in df.columns]
    if missing:
        sys.exit(f'ERROR: ESRB CSV missing columns: {missing}')
    df['title_norm'] = df['title'].apply(normalize_title)
    return df


def load_kaggle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1')

    # Normalise column names (dataset has some variation)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    col_map = {
        'name':             'title',
        'genre':            'genre',
        'platform':         'platform',
        'year_of_release':  'year_of_release',
        'publisher':        'publisher',
        'critic_score':     'critic_score',
        'user_score':       'user_score',
        'rating':           'kaggle_rating',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Numeric coercion
    df['critic_score']    = pd.to_numeric(df.get('critic_score'), errors='coerce')
    df['year_of_release'] = pd.to_numeric(df.get('year_of_release'), errors='coerce')

    # Clean user_score ("tbd" → NaN)
    if 'user_score' in df.columns:
        df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

    df['title_norm'] = df['title'].apply(normalize_title)
    print(f'  Kaggle dataset: {len(df):,} rows, '
          f'{df["critic_score"].notna().sum():,} with critic scores')
    return df


def merge_datasets(esrb: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join ESRB onto Kaggle on normalized title.
    Where multiple Kaggle rows match (same game, different platforms),
    keep the row with the highest critic score.
    """
    kaggle_dedup = (
        kaggle.sort_values('critic_score', ascending=False)
              .drop_duplicates(subset='title_norm', keep='first')
    [['title_norm','genre','platform','critic_score','year_of_release','publisher']]
    )

    merged = esrb.merge(kaggle_dedup, on='title_norm', how='left')
    matched = merged['critic_score'].notna().sum()
    print(f'  Merged: {len(merged):,} ESRB games, '
          f'{matched:,} matched to Kaggle ({matched/len(merged)*100:.1f}%)')
    return merged


def build_pipeline(algo_name: str) -> Pipeline:
    """Construct a full preprocessing + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('passthrough', 'passthrough', DESCRIPTOR_COLS + ENGINEERED_COLS),
            ('numeric',
             SimpleImputer(strategy='median'),
             NUMERIC_COLS),
            ('categorical',
             OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_COLS),
        ],
        remainder='drop',
    )

    base = ALGOS[algo_name]()

    inner = Pipeline([
        ('preprocessor', preprocessor),
        ('clf',          base),
    ])

    # Calibration makes confidence % statistically meaningful
    # (raw RF/GB probabilities are often overconfident)
    calibrated = CalibratedClassifierCV(inner, method='isotonic', cv=5)
    return calibrated


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train enriched ESRB v2 classifier')
    parser.add_argument('--esrb',   required=True,   help='Path to Video_games_esrb_rating.csv')
    parser.add_argument('--kaggle', required=True,   help='Path to Video_Games_Sales_with_Ratings.csv')
    parser.add_argument('--out',    default='models/esrb_v2.pkl', help='Output .pkl path')
    parser.add_argument('--algo',   default='gb',    choices=ALGOS.keys())
    parser.add_argument('--no-calibration', action='store_true',
                        help='Skip calibration (faster but less accurate probabilities)')
    args = parser.parse_args()

    print('\n── Step 1: Load datasets ─────────────────────────────────────')
    esrb   = load_esrb(args.esrb)
    kaggle = load_kaggle(args.kaggle)

    print('\n── Step 2: Merge ─────────────────────────────────────────────')
    df = merge_datasets(esrb, kaggle)

    # Fill missing categorical with placeholder
    df['genre']    = df['genre'].fillna('Unknown')
    df['platform'] = df['platform'].fillna('Unknown')

    print('\n── Step 3: Feature engineering ───────────────────────────────')
    df = engineer_features(df)
    print(f'  Total features: {len(ALL_FEATURES)}  '
          f'({len(DESCRIPTOR_COLS)} descriptors + '
          f'{len(ENGINEERED_COLS)} engineered + '
          f'{len(NUMERIC_COLS)} numeric + '
          f'{len(CATEGORICAL_COLS)} categorical)')

    X = df[ALL_FEATURES]
    y = df['esrb_rating'].values

    print(f'\n  Rating distribution:\n{pd.Series(y).value_counts().to_string()}')

    print('\n── Step 4: Train / test split ────────────────────────────────')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  Train: {len(X_train):,}   Test: {len(X_test):,}')

    print(f'\n── Step 5: Train ({args.algo.upper()} + calibration) ────────')
    model = build_pipeline(args.algo)
    if args.no_calibration:
        # Unwrap to just inner Pipeline
        model = model.estimator
    model.fit(X_train, y_train)

    print('\n── Step 6: Evaluate ──────────────────────────────────────────')
    y_pred = model.predict(X_test)
    print(f'\n  Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n')
    print(classification_report(y_test, y_pred))

    # Per-rating confidence check
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        classes = list(model.classes_) if hasattr(model,'classes_') else ['E','ET','M','T']
        print('\n  Average confidence per true rating:')
        for i, cls in enumerate(classes):
            mask = y_test == cls
            if mask.sum():
                avg_conf = proba[mask, i].mean() * 100
                print(f'    {cls:>3}: {avg_conf:.1f}%  (n={mask.sum()})')

    print('\n── Step 7: Save ──────────────────────────────────────────────')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(model.classes_) if hasattr(model,'classes_') else ['E','ET','M','T']

    payload = {
        # Core
        'model':               model,
        'classes':             classes,
        # Feature schema (used by model_loader.py to build input)
        'has_extra_features':  True,
        'descriptor_features': DESCRIPTOR_COLS,
        'engineered_features': ENGINEERED_COLS,
        'numeric_features':    NUMERIC_COLS,
        'categorical_features':CATEGORICAL_COLS,
        'all_features':        ALL_FEATURES,
        # Metadata
        'algo':                args.algo,
        'calibrated':          not args.no_calibration,
        'train_size':          len(X_train),
    }

    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f'\n  ✓ Saved to {out_path}')
    print('  Drop it in models/ and activate at /manage-model/\n')


if __name__ == '__main__':
    main()
