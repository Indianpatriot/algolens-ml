"""
Cross-validation sanity check - more honest accuracy estimate than a
single train/test split, especially given some classes have only 10-15
samples total.
Run from project root: python cv_check.py
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"])
y = df["label"]

clf = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
)

# 5-fold would leave only ~2 samples per fold for your smallest class
# (10 samples total) - use 3-fold so every fold has enough per class.
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

print("Per-fold accuracy:", scores)
print(f"Mean accuracy: {scores.mean():.4f}  (+/- {scores.std():.4f})")