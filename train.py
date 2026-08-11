"""
Phase 4: train a Random Forest classifier on data/features.csv.
Run from project root: python train.py
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Loaded {len(df)} samples, {X.shape[1]} features, {y.nunique()} classes")
    print(y.value_counts())
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("=== Test set accuracy ===")
    print(accuracy_score(y_test, preds))
    print()

    print("=== Classification report ===")
    print(classification_report(y_test, preds, zero_division=0))

    print("=== Confusion matrix ===")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    print("Labels order:", labels)
    print(cm)
    print()

    print("=== Feature importances (top 15) ===")
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(15))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_ORDER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved feature order to {FEATURE_ORDER_PATH}")


if __name__ == "__main__":
    main()