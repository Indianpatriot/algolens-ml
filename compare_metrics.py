import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("data/features.csv")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"])
y = df["label"]

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

affected = [
    "BFS",
    "Dynamic Programming",
    "Bucket Sort",
    "Recursive Non-Swap Sort",  # Merge Sort
    "Recursive Swap-Based Sort",  # Quick Sort
]

print("=== Classification Report (Affected Classes) ===")
rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
for cls in affected:
    metrics = rep.get(cls, {})
    prec = metrics.get("precision", 0)
    rec = metrics.get("recall", 0)
    f1 = metrics.get("f1-score", 0)
    support = metrics.get("support", 0)
    print(f"{cls:28s} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Support: {support}")

print("\n=== Confusion Matrix (Affected Classes) ===")
cm = confusion_matrix(y_test, preds, labels=affected)
print(f"{'True \\ Pred':28s} | " + " | ".join(f"{c[:10]:>10s}" for c in affected))
for i, cls in enumerate(affected):
    row_str = " | ".join(f"{cm[i][j]:10d}" for j in range(len(affected)))
    print(f"{cls:28s} | {row_str}")

