"""
Run this on YOUR machine (D:\\algolens-ml), from the project root.
Loads data/dataset.json, runs every sample through feature_extractor.py,
and writes data/features.csv with columns = [all features..., label].
Usage:
    python build_features.py
"""
import json
import csv
import shutil
import sys
from pathlib import Path

try:
    from feature_extractor import extract_features
except ImportError as e:
    print("Could not import extract_features from feature_extractor.py.")
    print("Make sure this script is in the same folder as feature_extractor.py,")
    print("and that the function is named extract_features(code: str) -> dict.")
    print(f"Original error: {e}")
    sys.exit(1)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "dataset.json"
    if not dataset_path.exists():
        legacy_path = base_dir / "algo-lens-viz" / "data" / "dataset.json"
        if legacy_path.exists():
            dataset_path = legacy_path
        else:
            raise FileNotFoundError(f"Could not find dataset.json at {dataset_path} or {legacy_path}")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    failed = []
    feature_names = None

    for i, sample in enumerate(dataset):
        code = sample["code"]
        label = sample["label"]
        try:
            features = extract_features(code)
        except Exception as e:
            failed.append((i, label, str(e)))
            continue

        if feature_names is None:
            feature_names = list(features.keys())

        row = dict(features)
        row["label"] = label
        rows.append(row)

    if failed:
        print(f"WARNING: {len(failed)} samples failed feature extraction:")
        for i, label, err in failed[:20]:
            print(f"  - sample #{i} ({label}): {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    if not rows:
        print("No features extracted - nothing to write. Fix the errors above first.")
        sys.exit(1)

    fieldnames = feature_names + ["label"]
    output_path = data_dir / "features.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}: {len(rows)} rows x {len(feature_names)} features + label")
    print(f"Feature columns: {feature_names}")
    print("\nSanity check: inspect data/features.csv and confirm classes are visually")
    print("separable, e.g. all BFS rows should have uses_queue=1, has_visited_like_var=1.")


if __name__ == "__main__":
    main()