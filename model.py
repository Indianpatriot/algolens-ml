"""
model.py - Classification engine with deterministic AST overrides and ML fallback.

Implements algorithm classification combining:
1. Deterministic AST signature rules (Quick Sort, Merge Sort, Insertion Sort,
   Shell Sort, Heap Sort, Counting/Radix/Bucket Sort, Binary Search, Union-Find,
   Monotonic Stack, Backtracking, etc.)
2. Trained Random Forest classifier fallback on structural AST feature vectors
   using SUB_LABELS to eliminate single-class majority bias.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import joblib
import pandas as pd

from feature_extractor import extract_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.pkl"

_clf = None
_feature_order: List[str] = []


def load_model(model_path: Optional[Path] = None, feature_order_path: Optional[Path] = None):
    """Loads and caches the trained ML classifier and feature ordering."""
    global _clf, _feature_order
    if _clf is None:
        m_path = model_path or MODEL_PATH
        f_path = feature_order_path or FEATURE_ORDER_PATH
        if not m_path.exists() or not f_path.exists():
            raise RuntimeError(
                f"Model files not found in {BASE_DIR}. Run train.py first."
            )
        _clf = joblib.load(m_path)
        _feature_order = joblib.load(f_path)
    return _clf, _feature_order


def apply_ast_rule_overrides(code: str, features: Dict[str, float]) -> Optional[Tuple[str, float, str]]:
    """
    Applies deterministic AST signature rules.
    Prioritizes unique structural signatures over default class predictions.
    
    Returns:
        (algorithm_name, confidence, reason) or None if no deterministic rule matches.
    """
    code_lower = code.lower()

    # 1. Quick Sort: Dual recursive calls + pivot partitioning / Lomuto-Hoare indices
    if features.get("has_dual_recursion", 0) == 1.0 and (
        features.get("has_pivot_partition", 0) == 1.0
        or features.get("has_lomuto_hoare_indexing", 0) == 1.0
        or "pivot" in code_lower
        or "partition" in code_lower
    ):
        return (
            "Quick Sort",
            0.99,
            "Detected dual recursive partitioning with pivot partitioning / Lomuto-Hoare index scheme.",
        )

    # 2. Merge Sort: Split slicing ([:mid], [mid:]) + parallel two-pointer / auxiliary merging
    if (features.get("has_split_slicing", 0) == 1.0 or features.get("has_dual_recursion", 0) == 1.0) and (
        features.get("has_parallel_two_pointer_pattern", 0) == 1.0
        or features.get("has_auxiliary_merging", 0) == 1.0
        or "merge" in code_lower
    ) and not (features.get("has_pivot_partition", 0) == 1.0 and "quick" in code_lower):
        return (
            "Merge Sort",
            0.99,
            "Detected divide-and-conquer split slicing with two-pointer auxiliary merging.",
        )

    # 3. Heap Sort: Child index math (2*i + 1, 2*i + 2) or heapify/sift helper
    if (features.get("has_heap_child_math", 0) == 1.0 or features.get("has_heapify_pattern", 0) == 1.0) and not features.get("has_pivot_partition", 0) and not features.get("uses_adjacency_structure", 0) and not features.get("has_visited_like_var", 0):
        return (
            "Heap Sort",
            0.99,
            "Detected binary heap restructure with child index math (2*i+1, 2*i+2) and heapify sift.",
        )

    # 4. Shell Sort: Shrinking gap sequence + element shifting / key assignment
    if features.get("has_gap_reduction", 0) == 1.0 and (
        features.get("has_element_shifting", 0) == 1.0
        or features.get("has_key_assignment", 0) == 1.0
        or features.get("has_swap_in_loop", 0) == 1.0
    ):
        return (
            "Shell Sort",
            0.98,
            "Detected variable shrinking gap sequence (gap //= 2) with array shifting/swapping.",
        )

    # 5. Insertion Sort: Key element assignment + shifting elements into sorted prefix
    if features.get("has_element_shifting", 0) == 1.0 and features.get("has_key_assignment", 0) == 1.0 and features.get("has_gap_reduction", 0) == 0.0:
        return (
            "Insertion Sort",
            0.99,
            "Detected key element assignment (key = arr[i]) and shifting (arr[j+1] = arr[j]).",
        )

    # 6. Radix Sort: Digit extraction + exponent scaling passes (exp *= 10)
    if features.get("has_radix_exp_scaling", 0) == 1.0 or ("exp" in code_lower and "radix" in code_lower):
        return (
            "Radix Sort",
            0.99,
            "Detected radix exponent scaling passes (exp *= 10) and digit bucket distribution.",
        )

    # 7. Counting Sort / Bucket Sort: Frequency / bucket allocation
    if (
        features.get("has_frequency_array", 0) == 1.0
        and features.get("has_radix_exp_scaling", 0) == 0.0
        and features.get("has_topological_sort_pattern", 0) == 0.0
        and features.get("has_prefix_pattern", 0) == 0.0
        and features.get("has_2d_list_indexing", 0) == 0.0
        and features.get("uses_queue", 0) == 0.0
    ):
        if "bucket" in code_lower:
            return (
                "Bucket Sort",
                0.98,
                "Detected bucket distribution list allocations and range partitioning.",
            )
        else:
            return (
                "Counting Sort",
                0.98,
                "Detected fixed-span frequency count array allocation ([0] * span).",
            )

    # 8. Binary Search: Midpoint halving calculation in single loop with no recursion
    if features.get("has_midpoint_calculation", 0) == 1.0 and features.get("max_loop_depth", 0) == 1.0 and features.get("has_recursion", 0) == 0.0:
        return (
            "Binary Search",
            0.99,
            "Detected midpoint-halving calculation ((lo+hi)//2) inside single search loop.",
        )

    # 9. Union-Find: Disjoint set find / union pattern
    if features.get("has_union_find_pattern", 0) == 1.0:
        return (
            "Union-Find",
            0.98,
            "Detected find/union disjoint set functions with parent tree tracking.",
        )

    # 10. Monotonic Stack: Conditional stack popping before pushing
    if features.get("has_monotonic_stack_pattern", 0) == 1.0:
        return (
            "Monotonic Stack",
            0.98,
            "Detected monotonic stack conditional pop-before-push loop structure.",
        )

    # 11. Backtracking: Recursive state explore-and-undo pattern
    if features.get("has_backtracking_undo_pattern", 0) == 1.0:
        return (
            "Backtracking",
            0.98,
            "Detected recursive path exploration with append-recurse-pop backtracking undo.",
        )

    # 12. Sliding Window: Arithmetic window resize in nested loop
    if features.get("has_sliding_window_pattern", 0) == 1.0:
        return (
            "Sliding Window",
            0.98,
            "Detected variable-length sliding window expansion and conditional shrink.",
        )

    # 13. Linked List Operations: List traversal / pointer mutation
    if (
        features.get("has_ll_insert_pattern", 0) == 1.0
        or features.get("has_ll_delete_pattern", 0) == 1.0
        or features.get("has_ll_reverse_pattern", 0) == 1.0
        or (features.get("has_linked_list_pattern", 0) == 1.0 and features.get("has_fast_slow_pointer_pattern", 0) == 0.0)
    ):
        return (
            "Linked List Operations",
            0.98,
            "Detected linked list pointer manipulations (.next traversal / mutations).",
        )

    # 14. Stack (LIFO): Sequential push / pop operations or stack structures
    if (
        (features.get("uses_stack", 0) == 1.0 or "stack" in code_lower or "stk" in code_lower)
        and ("pop(" in code_lower or "append(" in code_lower or "push(" in code_lower)
        and not features.get("has_visited_like_var", 0)
        and not features.get("uses_adjacency_structure", 0)
        and not features.get("has_backtracking_undo_pattern", 0)
    ):
        return (
            "Stack",
            0.99,
            "Detected Stack (LIFO) data structure operations (push/pop sequence).",
        )

    # 15. Queue (FIFO): Deque / queue operations (popleft, shift, pop(0))
    if (
        (features.get("uses_queue", 0) == 1.0 or "queue" in code_lower or "deque" in code_lower or "pop(0)" in code_lower or "popleft" in code_lower)
        and not features.get("has_visited_like_var", 0)
        and not features.get("uses_adjacency_structure", 0)
        and not features.get("has_topological_sort_pattern", 0)
    ):
        return (
            "Queue",
            0.99,
            "Detected Queue (FIFO) data structure operations (enqueue/dequeue sequence).",
        )

    return None


SUB_LABEL_TO_SPECIFIC = {
    "In-Place Nested-Loop Swap Sort": "Bubble Sort",
    "Shift-Based Sort": "Insertion Sort",
    "Recursive Non-Swap Sort": "Merge Sort",
    "Recursive Swap-Based Sort": "Quick Sort",
    "Heap/Complex Structural Sort": "Heap Sort",
    "Distribution Sort": "Counting Sort",
    "Single-Pass Loop Sort": "Gnome Sort",
}


def predict_algorithm(code: str, features: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
    """
    Predicts algorithm category using deterministic AST fallback rules first,
    falling back to the trained Random Forest classifier.
    
    Returns:
        (predicted_label, confidence_score)
    """
    if features is None:
        features = extract_features(code)

    # 1. Deterministic AST signature rules
    rule_match = apply_ast_rule_overrides(code, features)
    if rule_match is not None:
        pred_label, conf, _ = rule_match
        return pred_label, conf

    # 2. Fallback to trained Random Forest classifier
    clf, feature_order = load_model()
    row = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
    prediction = clf.predict(row)[0]
    confidence = float(clf.predict_proba(row).max())

    # Format generic sub-labels into specific template names
    if prediction in SUB_LABEL_TO_SPECIFIC:
        prediction = SUB_LABEL_TO_SPECIFIC[prediction]

    return prediction, round(confidence, 4)
