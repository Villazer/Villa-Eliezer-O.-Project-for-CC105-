"""
train_model_v1.py — Descriptor-only ESRB classifier (v1).

Uses only the 31 binary content descriptor columns from the ESRB CSV.
No external datasets required. Swap --algo to compare algorithms.

Usage
-----
  python train_model_v1.py \
      --data Video_games_esrb_rating.csv \
      --out  models/esrb_v1_gb.pkl \
      --algo gb
"""

import argparse, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

FEATURE_COLS = [
    'alcohol_reference','animated_blood','blood','blood_and_gore',
    'cartoon_violence','crude_humor','drug_reference','fantasy_violence',
    'intense_violence','language','lyrics','mature_humor','mild_blood',
    'mild_cartoon_violence','mild_fantasy_violence','mild_language',
    'mild_lyrics','mild_suggestive_themes','mild_violence','no_descriptors',
    'nudity','partial_nudity','sexual_content','sexual_themes',
    'simulated_gambling','strong_janguage','strong_sexual_content',
    'suggestive_themes','use_of_alcohol','use_of_drugs_and_alcohol','violence',
]

ALGOS = {
    'rf':  lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'gb':  lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, subsample=0.8, random_state=42),
    'lr':  lambda: LogisticRegression(max_iter=2000, class_weight='balanced'),
    'svm': lambda: SVC(probability=True, class_weight='balanced'),
}


def main():
    parser = argparse.ArgumentParser(description='Train descriptor-only ESRB v1 classifier')
    parser.add_argument('--data', required=True, help='Path to Video_games_esrb_rating.csv')
    parser.add_argument('--out',  default='models/esrb_v1.pkl')
    parser.add_argument('--algo', default='gb', choices=ALGOS.keys())
    parser.add_argument('--calibrate', action='store_true', default=True,
                        help='Wrap model in CalibratedClassifierCV (default: on)')
    parser.add_argument('--no-calibrate', dest='calibrate', action='store_false')
    args = parser.parse_args()

    print(f'\nLoading {args.data}…')
    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        sys.exit(f'Missing columns: {missing}')

    X = df[FEATURE_COLS].values
    y = df['esrb_rating'].values

    print(f'Dataset: {len(df):,} games  |  Rating distribution:')
    for r, n in pd.Series(y).value_counts().items():
        print(f'  {r}: {n}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'\nTraining {args.algo.upper()}{"+ calibration" if args.calibrate else ""}…')
    base = ALGOS[args.algo]()
    clf  = CalibratedClassifierCV(base, method='isotonic', cv=5) if args.calibrate else base
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n')
    print(classification_report(y_test, y_pred))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    classes = list(clf.classes_) if hasattr(clf, 'classes_') else ['E','ET','M','T']
    payload = {
        'model':              clf,
        'features':           FEATURE_COLS,
        'classes':            classes,
        'has_extra_features': False,
        'algo':               args.algo,
        'calibrated':         args.calibrate,
    }
    with open(out, 'wb') as f:
        pickle.dump(payload, f)

    print(f'\n✓ Saved to {out}')
    print('  Drop it in models/ and activate at /manage-model/\n')


if __name__ == '__main__':
    main()
