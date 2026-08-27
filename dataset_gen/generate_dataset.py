"""
AlgoLens - Phase 3 dataset generator.

Produces dataset.json: [{"code": "...", "label": "..."}, ...]

Categories BFS/DFS/DP/Backtracking/Two Pointer/Sliding Window/Binary
Search/Union-Find/Monotonic Stack each get one flat label per category.

Sorting is handled differently: it has 21 genuinely distinct algorithm
templates, many of which collapse to identical structural feature
signatures (e.g. Bubble/Selection/Comb/Cycle Sort all show
has_recursion=0, has_swap_in_loop=1, max_loop_depth=2 - indistinguishable
to feature_extractor.py no matter how much data you add). So Sorting
uses per-template SUB_LABELS (see templates_sorting.py) grouping the 21
algorithms into 7 structurally coherent sub-categories instead of one
flat 105-sample "Sorting" bucket.

Every generated sample is validated with ast.parse() before being kept -
if it doesn't parse, it can't go anywhere near feature_extractor.py.
"""

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from common import render, pick_func_name
import templates_bfs
import templates_dfs
import templates_dp
import templates_backtracking
import templates_two_pointer
import templates_sliding_window
import templates_binary_search
import templates_sorting
import templates_union_find
import templates_monotonic_stack
import templates_linked_list

# Flat-label categories: one label per whole templates list.
# Sorting is deliberately excluded here - handled separately below.
CATEGORIES = {
    "BFS": (templates_bfs.TEMPLATES, "bfs"),
    "DFS": (templates_dfs.TEMPLATES, "dfs"),
    "Dynamic Programming": (templates_dp.TEMPLATES, "dp"),
    "Backtracking": (templates_backtracking.TEMPLATES, "backtracking"),
    "Two Pointer": (templates_two_pointer.TEMPLATES, "two_pointer"),
    "Sliding Window": (templates_sliding_window.TEMPLATES, "sliding_window"),
    "Binary Search": (templates_binary_search.TEMPLATES, "binary_search"),
    "Union-Find": (templates_union_find.TEMPLATES, "union_find"),
    "Monotonic Stack": (templates_monotonic_stack.TEMPLATES, "monotonic_stack"),
    "Linked List Operations": (templates_linked_list.TEMPLATES, "linked_list"),
}

SORTING_FUNC_KEY = "sorting"
VARIANTS_PER_TEMPLATE = 8


def _generate_flat_categories():
    dataset = []
    stats = {}
    errors = []

    for label, (templates, func_key) in CATEGORIES.items():
        count_for_label = 0
        for t_idx, template in enumerate(templates):
            for variant in range(VARIANTS_PER_TEMPLATE):
                comment_on = (variant % 2 == 0)
                func_name = pick_func_name(func_key, variant + t_idx)
                code = render(template, variant + t_idx, comment_on, func_name=func_name)

                try:
                    tree = ast.parse(code)
                    ast.walk(tree)  # force full traversal
                except SyntaxError as e:
                    errors.append((label, t_idx, variant, str(e), code))
                    continue

                dataset.append({
                    "code": code,
                    "label": label,
                    "_source_template_idx": t_idx,
                    "_variant": variant,
                })
                count_for_label += 1
        stats[label] = count_for_label

    return dataset, stats, errors


def _generate_sorting_subcategories():
    """
    Sorting uses per-template SUB_LABELS instead of one flat label,
    since many of the 21 algorithms are structurally indistinguishable
    to feature_extractor.py (see module docstring above).
    """
    dataset = []
    stats = {}
    errors = []

    for t_idx, template in enumerate(templates_sorting.TEMPLATES):
        sub_label = templates_sorting.SUB_LABELS[t_idx]
        original_name = templates_sorting.TEMPLATE_NAMES[t_idx]

        for variant in range(VARIANTS_PER_TEMPLATE):
            comment_on = (variant % 2 == 0)
            func_name = pick_func_name(SORTING_FUNC_KEY, variant + t_idx)
            code = render(template, variant + t_idx, comment_on, func_name=func_name)

            try:
                tree = ast.parse(code)
                ast.walk(tree)
            except SyntaxError as e:
                errors.append((sub_label, t_idx, variant, str(e), code))
                continue

            dataset.append({
                "code": code,
                "label": sub_label,
                "_source_template_idx": t_idx,
                "_variant": variant,
                "_original_algorithm": original_name,
            })
            stats[sub_label] = stats.get(sub_label, 0) + 1

    return dataset, stats, errors


def generate():
    dataset, stats, errors = _generate_flat_categories()
    sort_dataset, sort_stats, sort_errors = _generate_sorting_subcategories()

    dataset.extend(sort_dataset)
    stats.update(sort_stats)
    errors.extend(sort_errors)

    return dataset, stats, errors


def main():
    dataset, stats, errors = generate()

    print("=== Generation report ===")
    total = 0
    for label, count in stats.items():
        print(f"  {label:35s} {count:3d} samples")
        total += count
    print(f"  {'TOTAL':35s} {total:3d} samples")

    if errors:
        print(f"\n!!! {len(errors)} samples FAILED to parse - fix templates before proceeding:")
        for label, t_idx, variant, err, code in errors:
            print(f"  - {label} template#{t_idx} variant#{variant}: {err}")
        sys.exit(1)

    # Write to the project-root-level data/ folder (sibling of dataset_gen/),
    # matching where build_features.py reads from.
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "dataset_debug.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    clean = [{"code": d["code"], "label": d["label"]} for d in dataset]
    with open(output_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    print(f"\nAll {total} samples parsed successfully.")
    print(f"Wrote dataset.json and dataset_debug.json to {output_dir}")


if __name__ == "__main__":
    main()