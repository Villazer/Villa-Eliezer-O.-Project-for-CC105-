"""
train_model.py — Standalone script to retrain the ESRB classifier.

Usage:
    python train_model.py --data path/to/Video_games_esrb_rating.csv \
                          --out  models/my_model.pkl \
                          --algo rf   # rf | gb | lr | svm

After saving, visit /manage-model/ in the browser to activate the new model.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

FEATURE_COLS = [
    'alcohol_reference', 'animated_blood', 'blood', 'blood_and_gore',
    'cartoon_violence', 'crude_humor', 'drug_reference', 'fantasy_violence',
    'intense_violence', 'language', 'lyrics', 'mature_humor', 'mild_blood',
    'mild_cartoon_violence', 'mild_fantasy_violence', 'mild_language',
    'mild_lyrics', 'mild_suggestive_themes', 'mild_violence', 'no_descriptors',
    'nudity', 'partial_nudity', 'sexual_content', 'sexual_themes',
    'simulated_gambling', 'strong_janguage', 'strong_sexual_content',
    'suggestive_themes', 'use_of_alcohol', 'use_of_drugs_and_alcohol', 'violence',
]

ALGOS = {
    'rf':  lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'gb':  lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
    'lr':  lambda: LogisticRegression(max_iter=1000, class_weight='balanced'),
    'svm': lambda: SVC(probability=True, class_weight='balanced'),
}


def main():
    parser = argparse.ArgumentParser(description='Train ESRB classifier')
    parser.add_argument('--data', required=True, help='Path to CSV file')
    parser.add_argument('--out',  default='models/esrb_model_.pkl', help='Output .pkl path')
    parser.add_argument('--algo', default='rf', choices=ALGOS.keys(), help='Algorithm')
    args = parser.parse_args()

    print(f'Loading data from {args.data}...')
    df = pd.read_csv(args.data)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f'ERROR: Missing columns: {missing}', file=sys.stderr)
        sys.exit(1)

    X = df[FEATURE_COLS].values
    y = df['esrb_rating'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'Training {args.algo} on {len(X_train)} samples...')
    clf = ALGOS[args.algo]()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'model':    clf,
        'features': FEATURE_COLS,
        'classes':  list(clf.classes_),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f'\n✓ Model saved to {out_path}')
    print('  Drop it in the models/ directory and activate it at /manage-model/')


if __name__ == '__main__':
    main()
