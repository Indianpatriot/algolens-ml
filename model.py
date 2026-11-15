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
import re
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
    if (
        (features.get("has_split_slicing", 0) == 1.0 or features.get("has_dual_recursion", 0) == 1.0)
        and (
            features.get("has_parallel_two_pointer_pattern", 0) == 1.0
            or features.get("has_auxiliary_merging", 0) == 1.0
            or "merge" in code_lower
        )
        and not (features.get("has_pivot_partition", 0) == 1.0 and "quick" in code_lower)
        and features.get("has_backtracking_undo_pattern", 0) == 0.0
    ):
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

    # 5. Insertion Sort: Key element assignment + shifting elements into sorted prefix (without bucket distribution)
    if (
        features.get("has_element_shifting", 0) == 1.0
        and features.get("has_key_assignment", 0) == 1.0
        and features.get("has_gap_reduction", 0) == 0.0
        and features.get("has_bucket_comprehension", 0) == 0.0
        and features.get("has_bucket_index_calculation", 0) == 0.0
        and features.get("has_bucket_concatenation", 0) == 0.0
    ):
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

    # 7. Bucket Sort: list of lists comprehension + index calculation + bucket sort/concat loops
    if (
        (features.get("has_bucket_comprehension", 0) == 1.0 and (features.get("has_bucket_index_calculation", 0) == 1.0 or features.get("has_bucket_sort_loop", 0) == 1.0 or features.get("has_bucket_concatenation", 0) == 1.0))
        or (features.get("has_bucket_index_calculation", 0) == 1.0 and (features.get("has_bucket_sort_loop", 0) == 1.0 or features.get("has_bucket_concatenation", 0) == 1.0))
        or ("bucket" in code_lower and (features.get("has_bucket_comprehension", 0) == 1.0 or features.get("has_frequency_array", 0) == 1.0 or "sort" in code_lower))
    ) and not features.get("has_topological_sort_pattern", 0) and not features.get("has_2d_list_indexing", 0):
        return (
            "Bucket Sort",
            0.99,
            "Detected bucket distribution list allocations, range partitioning, and per-bucket sorting/concatenation.",
        )

    # 7b. Counting Sort: Frequency array allocation + element-indexed incrementation / output rebuild
    if (
        (
            features.get("has_element_indexed_count", 0) == 1.0
            or features.get("has_count_sort_rebuild", 0) == 1.0
            or (features.get("has_frequency_array", 0) == 1.0 and ("count" in code_lower or "freq" in code_lower or "counting" in code_lower))
        )
        and features.get("has_word_break_pattern", 0) == 0.0
        and features.get("has_coin_change_ways_pattern", 0) == 0.0
        and features.get("has_coin_change_min_pattern", 0) == 0.0
        and features.get("has_knapsack_01_pattern", 0) == 0.0
        and features.get("has_earlier_dp_dependency", 0) == 0.0
        and features.get("has_string_slicing_subscript", 0) == 0.0
        and features.get("has_radix_exp_scaling", 0) == 0.0
        and features.get("has_topological_sort_pattern", 0) == 0.0
        and features.get("has_prefix_pattern", 0) == 0.0
        and features.get("has_2d_list_indexing", 0) == 0.0
        and features.get("uses_queue", 0) == 0.0
        and features.get("has_bucket_comprehension", 0) == 0.0
        and features.get("has_bucket_index_calculation", 0) == 0.0
    ):
        return (
            "Counting Sort",
            0.98,
            "Detected fixed-span frequency count array allocation ([0] * span) and element-indexed incrementation.",
        )

    # 7c. All-Pairs Shortest Path (repeated Bellman-Ford): outer loop over sources running SSSP
    if (
        features.get("has_apsp_nested_sources", 0) == 1.0
        or ("apsp" in code_lower and ("bellman" in code_lower or "dist" in code_lower))
        or (
            features.get("has_bellman_ford_edges_loop", 0) == 1.0
            and ("for src in" in code_lower or "for u in range" in code_lower or "matrix" in code_lower)
        )
    ):
        return (
            "All-Pairs Shortest Path (repeated Bellman-Ford)",
            0.99,
            "Detected All-Pairs Shortest Path running repeated Bellman-Ford SSSP passes across all source vertices.",
        )

    # 7d. DAG Shortest Path: topological ordering followed by edge relaxation
    if (
        features.get("has_dag_relaxation_loop", 0) == 1.0
        or (
            (features.get("has_topological_sort_pattern", 0) == 1.0 or features.get("has_topological_sort_kahn", 0) == 1.0 or "topo" in code_lower)
            and (features.get("has_dist_relaxation", 0) == 1.0 or "dist[" in code_lower)
            and not features.get("uses_priority_queue", 0)
            and "heappop" not in code_lower
        )
    ):
        return (
            "DAG Shortest Path",
            0.99,
            "Detected DAG Shortest Path: linear-time single-source shortest path using topological ordering followed by edge relaxation.",
        )

    # 7e. Bellman-Ford: V-1 edge relaxation passes with negative cycle check
    if (
        features.get("has_bellman_ford_edges_loop", 0) == 1.0
        or features.get("has_negative_cycle_check", 0) == 1.0
        or ("bellman" in code_lower and ("edges" in code_lower or "dist" in code_lower or "negative" in code_lower))
    ):
        return (
            "Bellman-Ford",
            0.99,
            "Detected Bellman-Ford single-source shortest path with V-1 edge relaxation passes and negative cycle detection.",
        )

    # 7f. 0-1 BFS: deque frontier pushing 0-weight edges to front and 1-weight edges to back
    if (
        features.get("has_deque_appendleft", 0) == 1.0
        or (features.get("has_zero_one_weights", 0) == 1.0 and (features.get("uses_queue", 0) == 1.0 or "deque" in code_lower))
        or ("appendleft" in code_lower and ("weight" in code_lower or "dist" in code_lower))
    ):
        return (
            "0-1 BFS",
            0.99,
            "Detected 0-1 BFS with double-ended queue (deque) pushing 0-weight edges to front and 1-weight edges to back.",
        )

    # 7g. Dijkstra's Algorithm (Single-Source Shortest Path): Priority queue + distance relaxation
    if (
        (
            features.get("has_priority_queue_traversal", 0) == 1.0
            or features.get("uses_priority_queue", 0) == 1.0
            or "heappop" in code_lower
        )
        and (
            features.get("has_dist_relaxation", 0) == 1.0
            or "dijkstra" in code_lower
            or ("dist" in code_lower and ("< dist[" in code_lower or "+ weight" in code_lower or "+ w" in code_lower))
        )
        and features.get("has_mst_weight_accumulation", 0) == 0.0
    ):
        return (
            "Dijkstra's Algorithm",
            0.99,
            "Detected Dijkstra's SSSP algorithm (min-heap priority queue with edge distance relaxation).",
        )

    # 7h. Prim's Algorithm (Minimum Spanning Tree): Priority queue + cut set without distance relaxation
    if (
        (
            features.get("has_priority_queue_traversal", 0) == 1.0
            or features.get("uses_priority_queue", 0) == 1.0
            or "heappop" in code_lower
            or "heappush" in code_lower
        )
        and (
            features.get("has_mst_weight_accumulation", 0) == 1.0
            or features.get("has_mst_edge_building", 0) == 1.0
            or (features.get("has_weighted_adjacency", 0) == 1.0 and features.get("has_lazy_deletion_on_pop", 0) == 1.0)
            or ("prim" in code_lower and ("graph" in code_lower or "adj" in code_lower or "mst" in code_lower))
            or ("mst" in code_lower and ("visited" in code_lower or "seen" in code_lower or "total" in code_lower))
        )
        and features.get("has_dist_relaxation", 0) == 0.0
        and not any(k in code_lower for k in ("dijkstra", "shortest_path"))
    ):
        return (
            "Prim's Algorithm",
            0.99,
            "Detected Prim's MST algorithm (min-heap priority queue with visited cut set, edge weight selection, and lazy deletion).",
        )

    # 7i. Topological Sort (Kahn's algorithm)
    if (
        features.get("has_topological_sort_kahn", 0) == 1.0
        or (
            features.get("has_topological_sort_pattern", 0) == 1.0
            and (features.get("uses_queue", 0) == 1.0 or "in_degree" in code_lower or "indegree" in code_lower)
        )
    ) and features.get("has_dag_relaxation_loop", 0) == 0.0:
        return (
            "Topological Sort (Kahn)",
            0.99,
            "Detected Kahn's algorithm for topological sorting using in-degree tracking and queue processing.",
        )

    # 7j. Topological Sort (DFS post-order with cycle detection)
    if (
        features.get("has_topological_sort_dfs", 0) == 1.0
        or features.get("has_cycle_path_set", 0) == 1.0
        or (
            features.get("has_topo_dfs_postorder", 0) == 1.0
            and (features.get("has_reversed_stack_return", 0) == 1.0 or features.get("has_three_color_cycle_detection", 0) == 1.0)
        )
    ) and features.get("has_dag_relaxation_loop", 0) == 0.0 and not features.get("has_dist_relaxation", 0):
        return (
            "Topological Sort (DFS)",
            0.99,
            "Detected DFS-based topological sort with cycle detection and post-order result reversal.",
        )

    # 7k. Topological Sort generic fallback
    if (
        (
            features.get("has_topological_sort_pattern", 0) == 1.0
            or ("topological" in code_lower or "topo_sort" in code_lower or "toposort" in code_lower)
        )
        and features.get("has_dp_table_indexing", 0) == 0.0
        and features.get("has_priority_queue_traversal", 0) == 0.0
        and features.get("has_dag_relaxation_loop", 0) == 0.0
    ):
        return (
            "Topological Sort",
            0.99,
            "Detected DAG topological sort with cycle detection, in-degree/DFS post-order ordering.",
        )

    # 7l. DFS (iterative): stack + reversed neighbor iteration or stack.extend
    if (
        (
            features.get("has_reversed_neighbor_traversal", 0) == 1.0
            or features.get("has_stack_extend_neighbors", 0) == 1.0
            or (
                (features.get("uses_stack", 0) == 1.0 or "stack" in code_lower or "stk" in code_lower)
                and (features.get("uses_adjacency_structure", 0) == 1.0 or features.get("has_adjacency_iteration", 0) == 1.0)
                and ("pop(" in code_lower or ".pop()" in code_lower)
            )
        )
        and features.get("has_topological_sort_pattern", 0) == 0.0
        and features.get("has_topological_sort_kahn", 0) == 0.0
        and features.get("has_topological_sort_dfs", 0) == 0.0
        and features.get("has_dag_relaxation_loop", 0) == 0.0
        and features.get("has_dist_relaxation", 0) == 0.0
        and features.get("has_backtracking_undo_pattern", 0) == 0.0
        and features.get("uses_queue", 0) == 0.0
    ):
        return (
            "DFS (iterative)",
            0.99,
            "Detected iterative DFS graph traversal with LIFO stack frontier and neighbor traversal.",
        )

    # 7m. DFS (recursive): recursion over graph adjacency with visited state
    if (
        features.get("has_recursion", 0) == 1.0
        and (features.get("uses_adjacency_structure", 0) == 1.0 or features.get("has_adjacency_iteration", 0) == 1.0)
        and (features.get("has_visited_like_var", 0) == 1.0 or features.get("uses_set", 0) == 1.0 or "visited" in code_lower)
        and features.get("has_backtracking_undo_pattern", 0) == 0.0
        and features.get("has_topological_sort_pattern", 0) == 0.0
        and features.get("has_topological_sort_dfs", 0) == 0.0
        and features.get("has_dag_relaxation_loop", 0) == 0.0
        and features.get("has_dist_relaxation", 0) == 0.0
        and not any(k in code_lower for k in ("n_queens", "permute", "combination_sum", "word_search"))
    ):
        return (
            "DFS",
            0.99,
            "Detected recursive depth-first search graph traversal with visited state tracking.",
        )

    # 7n. BFS (Breadth-First Search): Queue frontier exploration + visited set tracking
    if (
        (features.get("has_queue_while_loop", 0) == 1.0 or features.get("uses_queue", 0) == 1.0 or "deque" in code_lower or "queue" in code_lower)
        and (features.get("has_queue_pop", 0) == 1.0 or "popleft" in code_lower or "pop(0)" in code_lower)
        and (
            features.get("has_visited_check_before_enqueue", 0) == 1.0
            or features.get("has_adjacency_iteration", 0) == 1.0
            or features.get("uses_adjacency_structure", 0) == 1.0
            or features.get("has_visited_like_var", 0) == 1.0
            or features.get("uses_set", 0) == 1.0
        )
        and features.get("has_dp_table_indexing", 0) == 0.0
        and features.get("has_memo_dict_access", 0) == 0.0
        and features.get("has_topological_sort_pattern", 0) == 0.0
        and features.get("has_topological_sort_kahn", 0) == 0.0
        and features.get("has_priority_queue_traversal", 0) == 0.0
        and features.get("uses_priority_queue", 0) == 0.0
        and features.get("has_deque_appendleft", 0) == 0.0
        and features.get("has_dist_relaxation", 0) == 0.0
        and "heappop" not in code_lower
        and "heappush" not in code_lower
    ):
        return (
            "BFS",
            0.99,
            "Detected BFS queue frontier exploration with visited state tracking and adjacency iteration.",
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
    if features.get("has_backtracking_undo_pattern", 0) == 1.0 or any(k in code_lower for k in ("n_queens", "nqueens", "queens", "sudoku", "permute", "combination_sum", "word_search", "exist")):
        if any(k in code_lower for k in ("n_queens", "nqueens", "queens")):
            return (
                "N-Queens",
                0.99,
                "Detected N-Queens board backtracking placement with conflict checks and un-placement.",
            )
        if any(k in code_lower for k in ("permute", "permutations")):
            return (
                "Permutations",
                0.99,
                "Detected permutation state exploration with recursive branching and backtracking undo.",
            )
        if any(k in code_lower for k in ("word_search", "exist", "word search")):
            return (
                "Word Search",
                0.99,
                "Detected 2D grid DFS word search with character matching, cell marking, and backtracking unmark.",
            )
        return (
            "Backtracking",
            0.98,
            "Detected recursive path exploration with state mutation, recursion, and backtracking undo.",
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
        and not any(k in code_lower for k in ("level", "root", ".left", ".right", "tree", "bfs"))
    ):
        return (
            "Queue",
            0.99,
            "Detected Queue (FIFO) data structure operations (enqueue/dequeue sequence).",
        )

    # 16. 3Sum (Two Pointers): Sorted array + outer loop + converging two pointers
    if (
        ("three_sum" in code_lower or "threesum" in code_lower or "3sum" in code_lower)
        or (
            ("sort()" in code_lower or "sorted(" in code_lower)
            and ("left" in code_lower and "right" in code_lower)
            and ("left < right" in code_lower or "l < r" in code_lower)
            and ("+ nums[" in code_lower or "+ arr[" in code_lower or "total" in code_lower or "sum" in code_lower)
        )
    ):
        return (
            "3Sum",
            0.99,
            "Detected 3Sum two-pointer pattern (sorted preprocessing with fixed outer loop and converging inner two pointers).",
        )

    # 16b. Two Pointer (Converging or Same-Direction):
    if (
        (
            features.get("has_two_pointer_pattern", 0) == 1.0
            or features.get("has_same_direction_pointer_pattern", 0) == 1.0
            or (
                ("left" in code_lower and "right" in code_lower and ("<" in code_lower or "<=" in code_lower))
                and not features.get("has_midpoint_calculation", 0)
                and not features.get("has_recursion", 0)
            )
            or ("read" in code_lower and "write" in code_lower)
            or any(k in code_lower for k in ("two_sum", "twosum", "two-sum", "remove_duplicates", "removeduplicates", "remove_element", "move_zeroes"))
        )
        and not features.get("has_midpoint_calculation", 0)
        and not features.get("has_recursion", 0)
        and not features.get("has_linked_list_pattern", 0)
        and not features.get("has_sliding_window_pattern", 0)
    ):
        return (
            "Two Pointer",
            0.99,
            "Detected two-pointer traversal (converging or same-direction pointer manipulation on array).",
        )

    # 17. Prim's Algorithm (Minimum Spanning Tree): Priority queue / min-heap + edge weights + cut set without distance relaxation
    if (
        (
            features.get("has_priority_queue_traversal", 0) == 1.0
            or features.get("uses_priority_queue", 0) == 1.0
            or "heappop" in code_lower
            or "heappush" in code_lower
        )
        and (
            features.get("has_mst_weight_accumulation", 0) == 1.0
            or features.get("has_mst_edge_building", 0) == 1.0
            or (features.get("has_weighted_adjacency", 0) == 1.0 and features.get("has_lazy_deletion_on_pop", 0) == 1.0)
            or ("prim" in code_lower and ("graph" in code_lower or "adj" in code_lower or "mst" in code_lower))
            or ("mst" in code_lower and ("visited" in code_lower or "seen" in code_lower or "total" in code_lower))
        )
        and features.get("has_dist_relaxation", 0) == 0.0
        and not any(k in code_lower for k in ("dijkstra", "shortest_path"))
    ):
        return (
            "Prim's Algorithm",
            0.99,
            "Detected Prim's MST algorithm (min-heap priority queue with visited cut set, edge weight selection, and lazy deletion).",
        )

    # 17b. Dijkstra's Algorithm (Single-Source Shortest Path): Priority queue + distance relaxation
    if (
        (
            features.get("has_priority_queue_traversal", 0) == 1.0
            or features.get("uses_priority_queue", 0) == 1.0
            or "heappop" in code_lower
        )
        and (
            features.get("has_dist_relaxation", 0) == 1.0
            or "dijkstra" in code_lower
        )
        and features.get("has_mst_weight_accumulation", 0) == 0.0
    ):
        return (
            "Dijkstra's Algorithm",
            0.99,
            "Detected Dijkstra's SSSP algorithm (min-heap priority queue with edge distance relaxation).",
        )

    # 18. Array Intersection (Hash Set): Set initialization + membership checking
    if (
        ("set(" in code_lower or "new set" in code_lower)
        and ("in set" in code_lower or ".has(" in code_lower or ".remove(" in code_lower or "set1" in code_lower)
        and ("intersection" in code_lower or "intersect" in code_lower or ("nums1" in code_lower and "nums2" in code_lower))
    ):
        return (
            "Array Intersection",
            0.99,
            "Detected Array Intersection using Hash Set (set creation and element membership checking).",
        )

    # 18b. Greedy / Interval Scheduling: Earliest finish time sorting + single pass compatibility scan
    if (
        features.get("has_interval_scheduling_pattern", 0) == 1.0
        or (features.get("has_interval_sort_key", 0) == 1.0 and features.get("has_interval_compatibility_check", 0) == 1.0)
        or (
            ("interval" in code_lower)
            and ("sort" in code_lower or "sorted" in code_lower)
            and ("end" in code_lower or "last" in code_lower or "finish" in code_lower or "prev" in code_lower)
            and not features.get("has_dp_table_indexing", 0)
        )
    ) and not features.get("has_dp_table_indexing", 0) and not features.get("has_memo_dict_access", 0):
        return (
            "Greedy / Interval Scheduling",
            0.99,
            "Detected greedy interval scheduling with earliest finish time sorting and single-pass selection.",
        )

    # 19. Dynamic Programming:
    # 1D or 2D table recurrence transitions (dp[i] = min/max(dp[i - c]), dp[i][j] = ...), or memoization
    has_dp_name = bool(re.search(r'\b(?:coinchange|coin_change|climbstairs|climb_stairs|knapsack|longestcommonsubsequence|robber|lis|wordbreak|word_break)\b', code_lower))
    if (
        features.get("has_dp_table_indexing", 0) == 1.0
        or features.get("has_memoization", 0) == 1.0
        or features.get("has_memo_dict_access", 0) == 1.0
        or has_dp_name
        or (
            ("dp[" in code_lower or "table[" in code_lower)
            and ("min(" in code_lower or "max(" in code_lower or "+" in code_lower)
            and not features.get("has_backtracking_undo_pattern", 0)
            and not features.get("has_topological_sort_pattern", 0)
            and not features.get("has_topo_dfs_postorder", 0)
            and not features.get("has_interval_scheduling_pattern", 0)
        )
    ) and (
        not features.get("has_backtracking_undo_pattern", 0)
        and not features.get("has_topological_sort_pattern", 0)
        and not features.get("has_topo_dfs_postorder", 0)
        and not features.get("has_interval_scheduling_pattern", 0)
        and not features.get("has_interval_sort_key", 0)
    ):
        # Fine-grained DP pattern identification
        if features.get("has_coin_change_min_pattern", 0) == 1.0 or (("coin" in code_lower or "coinchange" in code_lower or "coin_change" in code_lower or "fewest" in code_lower) and features.get("has_1d_dp_table", 0) == 1.0 and "min" in code_lower):
            return (
                "Dynamic Programming (Coin Change)",
                0.99,
                "Detected 1D Dynamic Programming: Coin Change (Minimum Coins) with recurrence dp[x] = min(dp[x], dp[x - coin] + 1).",
            )
        if features.get("has_coin_change_ways_pattern", 0) == 1.0 or (bool(re.search(r'\b(?:change|ways|combinations)\b', code_lower)) and "coin" in code_lower and features.get("has_1d_dp_table", 0) == 1.0):
            return (
                "Dynamic Programming (Coin Change - Number of Ways)",
                0.99,
                "Detected 1D Dynamic Programming: Coin Change (Number of Ways) with recurrence dp[x] += dp[x - coin].",
            )
        if "subsequence" in code_lower or "lcs" in code_lower:
            return (
                "Dynamic Programming (LCS)",
                0.99,
                "Detected 2D Dynamic Programming: Longest Common Subsequence.",
            )
        if features.get("has_knapsack_01_pattern", 0) == 1.0 or ("knapsack" in code_lower and (features.get("has_2d_dp_table", 0) == 1.0 or features.get("has_2d_list_indexing", 0) == 1.0 or "val" in code_lower)):
            return (
                "Dynamic Programming (0/1 Knapsack)",
                0.99,
                "Detected 2D Dynamic Programming: 0/1 Knapsack with inclusion-exclusion recurrence.",
            )
        if "climb" in code_lower or "stairs" in code_lower:
            return (
                "Dynamic Programming (Climbing Stairs)",
                0.99,
                "Detected 1D Dynamic Programming: Climbing Stairs with Fibonacci-style counting recurrence.",
            )

        return (
            "Dynamic Programming",
            0.99,
            "Detected dynamic programming recurrence relation with state table transitions or memoization.",
        )

    # 19b. Word Break / String Partition DP: dp array with string slicing & dictionary lookup
    if (
        features.get("has_word_break_pattern", 0) == 1.0
        or (
            features.get("has_string_slicing_subscript", 0) == 1.0
            and features.get("has_earlier_dp_dependency", 0) == 1.0
        )
        or (
            ("word_break" in code_lower or "wordbreak" in code_lower)
            and (features.get("has_string_slicing_subscript", 0) == 1.0 or "dp" in code_lower or "dict" in code_lower or "word" in code_lower)
        )
    ):
        return (
            "Dynamic Programming",
            0.99,
            "Detected Word Break string partition dynamic programming (dp[i] = dp[j] and s[j:i] in word_set).",
        )

    return None


SUB_LABEL_TO_SPECIFIC = {
    "In-Place Nested-Loop Swap Sort": "Bubble Sort",
    "Shift-Based Sort": "Insertion Sort",
    "Recursive Non-Swap Sort": "Merge Sort",
    "Recursive Swap-Based Sort": "Quick Sort",
    "Heap/Complex Structural Sort": "Heap Sort",
    "Distribution Sort": "Counting Sort",
    "Bucket Sort": "Bucket Sort",
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
    
    probs = clf.predict_proba(row)[0]
    classes = clf.classes_
    sorted_idx = probs.argsort()[::-1]
    top1_idx = sorted_idx[0]
    top2_idx = sorted_idx[1] if len(sorted_idx) > 1 else None

    prediction = classes[top1_idx]
    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx]) if top2_idx is not None else 0.0
    margin = top1_prob - top2_prob

    # Structural contradiction checks:
    # If structural features indicate graph adjacency but ML predicts sorting/DP without graph structures
    is_graph_code = features.get("uses_adjacency_structure", 0) == 1.0 or features.get("has_adjacency_iteration", 0) == 1.0
    if is_graph_code and prediction not in (
        "BFS",
        "DFS",
        "DFS (iterative)",
        "Topological Sort",
        "Topological Sort (Kahn)",
        "Topological Sort (DFS)",
        "Bellman-Ford",
        "0-1 BFS",
        "All-Pairs Shortest Path (repeated Bellman-Ford)",
        "DAG Shortest Path",
        "Dijkstra's Algorithm",
        "Prim's Algorithm",
        "Union-Find",
    ):
        return "Uncertain", round(top1_prob, 4)

    # If structural features indicate intervals/greedy but ML predicts DP
    if (features.get("has_interval_sort_key", 0) == 1.0 or features.get("has_interval_compatibility_check", 0) == 1.0) and prediction == "Dynamic Programming":
        return "Uncertain", round(top1_prob, 4)

    # Confidence calibration: if margin is too narrow (< 0.10) or top-1 probability is low (< 0.40)
    CONFIDENCE_FLOOR = 0.40
    MARGIN_FLOOR = 0.10
    if top1_prob < CONFIDENCE_FLOOR or (margin < MARGIN_FLOOR and top1_prob < 0.55):
        return "Uncertain", round(top1_prob, 4)

    # Format generic sub-labels into specific template names
    if prediction in SUB_LABEL_TO_SPECIFIC:
        prediction = SUB_LABEL_TO_SPECIFIC[prediction]

    return prediction, round(top1_prob, 4)
