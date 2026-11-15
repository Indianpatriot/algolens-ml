import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def main():
    with open("data/dataset.json", "r") as f:
        data = json.load(f)

    df_feat = pd.read_csv("data/features.csv")
    X = df_feat.drop(columns=["label"]).values
    y = df_feat["label"].values

    model = joblib.load("model.pkl")
    preds = model.predict(X)

    classes = ["Dynamic Programming", "Sorting", "BFS", "Greedy / Interval Scheduling", "Topological Sort"]
    print("=== MODEL EVALUATION ON FULL DATASET (720 samples) ===")
    print(classification_report(y, preds, zero_division=0))

    target_labels = ["BFS", "Dynamic Programming", "Sorting", "Greedy / Interval Scheduling", "Topological Sort"]
    cm = confusion_matrix(y, preds, labels=target_labels)
    print("\nConfusion Matrix for Target Classes:")
    print("Labels:", target_labels)
    for row_label, row in zip(target_labels, cm):
        print(f"{row_label:30}: {row}")

if __name__ == "__main__":
    main()
