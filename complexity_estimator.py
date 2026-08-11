"""
complexity_estimator.py

Rule-based (NOT machine-learned) time/space complexity estimation.
Uses the exact same 36-feature vector produced by feature_extractor.py,
but reasons about it deterministically rather than statistically - this
is a genuinely different kind of module from train.py's classifier, and
that's a deliberate design choice: complexity analysis has known, well-
defined rules (nested-loop depth, recursion shape, etc.), so encoding
those directly is more honest and more reliable than trying to have a
model "learn" complexity classes from limited data.

Rules are checked in order, most-specific-and-most-confident first, with
a series of fallbacks. The first matching rule wins. Every rule returns
WHY it matched (the specific features that triggered it), not just the
final answer - this reasoning is what gets surfaced in the "Complexity
Report" tab of the frontend.

Known limitation (worth stating explicitly, not hiding): static feature-
based estimation cannot always distinguish between algorithms with
identical structural shape but different complexity guarantees (e.g.
average-case vs worst-case Quick Sort, or an optimized vs naive DP
recurrence). Where this ambiguity exists, the estimator says so in its
reasoning rather than picking one answer with false confidence.
"""

from typing import Dict
from feature_extractor import extract_features


class ComplexityEstimate:
    def __init__(self, time: str, space: str, reasoning: str, confidence: str = "high"):
        self.time = time
        self.space = space
        self.reasoning = reasoning
        self.confidence = confidence  # "high" | "medium" | "low"

    def to_dict(self) -> Dict[str, str]:
        return {
            "time_complexity": self.time,
            "space_complexity": self.space,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


def _rule_binary_search(f: dict):
    """Midpoint-halving search: O(log n) time, O(1) space (iterative)."""
    if f["has_midpoint_calculation"] and f["max_loop_depth"] == 1 and not f["has_recursion"]:
        return ComplexityEstimate(
            time="O(log n)",
            space="O(1)",
            reasoning=(
                "Detected a midpoint calculation ((lo+hi)//2 style) inside a single "
                "loop with no recursion - the classic halving-search shape. Each "
                "iteration eliminates half the remaining search space."
            ),
            confidence="high",
        )
    return None


def _rule_union_find(f: dict):
    """Union-Find with path compression: near O(1) amortized per operation."""
    if f["has_union_find_pattern"]:
        return ComplexityEstimate(
            time="O(alpha(n)) amortized per operation",
            space="O(n)",
            reasoning=(
                "Detected the find/union + parent-array signature. With path "
                "compression (recursive find that rewrites parent pointers), each "
                "operation is nearly constant time on average - technically bounded "
                "by the inverse Ackermann function alpha(n), which is effectively "
                "constant for any n that could exist in practice. Space is O(n) for "
                "the parent array."
            ),
            confidence="medium",
        )
    return None


def _rule_recursive_memoized(f: dict):
    """Recursion + memoization: complexity depends on the state space size,
    not the raw branching factor - memoization collapses repeated subproblems."""
    if f["has_recursion"] and f["has_memoization"]:
        if f["array_dimensionality"] >= 2:
            return ComplexityEstimate(
                time="O(n * m)",
                space="O(n * m)",
                reasoning=(
                    "Recursive function with memoization, and the memo structure is "
                    "2-dimensional (dp[i][j]-style). Each distinct (i, j) state is "
                    "computed once and cached, so total work is bounded by the number "
                    "of distinct states - here, the product of the two dimensions."
                ),
                confidence="medium",
            )
        return ComplexityEstimate(
            time="O(n)",
            space="O(n)",
            reasoning=(
                "Recursive function with memoization (results cached in a dict/array "
                "keyed by a single parameter). Each distinct input value is computed "
                "once; the memoization collapses what would otherwise be exponential "
                "repeated recursive calls down to linear work."
            ),
            confidence="medium",
        )
    return None


def _rule_backtracking(f: dict):
    """Backtracking (append -> recurse -> undo): exponential search space."""
    if f["has_backtracking_undo_pattern"]:
        return ComplexityEstimate(
            time="O(k^n) or O(n!) (problem-dependent, exponential)",
            space="O(n)",
            reasoning=(
                "Detected the append/recurse/undo backtracking shape. Backtracking "
                "explores a branching search tree with pruning; the exact exponential "
                "base depends on the branching factor of the specific problem (e.g. "
                "O(n!) for permutation-style search, O(2^n) for subset-style search). "
                "Space is O(n) for the recursion depth / partial-solution path."
            ),
            confidence="low",
        )
    return None


def _rule_recursive_unmemoized_branching(f: dict):
    """Recursion with 2+ branches, no memoization, no swap/merge signature:
    the classic naive-Fibonacci-style exponential blowup."""
    if (
        f["has_recursion"]
        and not f["has_memoization"]
        and f["recursion_branch_count"] >= 2
        and not f["has_swap_in_loop"]
        and not f["has_parallel_two_pointer_pattern"]
        and not f["has_tree_pattern"]
    ):
        return ComplexityEstimate(
            time="O(2^n)",
            space="O(n)",
            reasoning=(
                "Recursive function calling itself 2+ times per invocation, with no "
                "memoization and no divide-and-conquer signature (no merge step, no "
                "in-place partition). This is the naive-recursion shape where the same "
                "subproblems get recomputed repeatedly, causing exponential blowup. "
                "Space is O(n) for the recursion call stack depth."
            ),
            confidence="medium",
        )
    return None


def _rule_divide_and_conquer_merge(f: dict):
    """Recursion + 2 branches + parallel-two-pointer merge step: Merge-Sort-style."""
    if f["has_recursion"] and f["recursion_branch_count"] >= 2 and f["has_parallel_two_pointer_pattern"]:
        return ComplexityEstimate(
            time="O(n log n)",
            space="O(n)",
            reasoning=(
                "Recursive function that splits into 2 recursive calls and then "
                "merges results with a two-index parallel scan - the classic merge-"
                "sort divide-and-conquer shape. The problem is halved at each "
                "recursion level (log n levels), and merging at each level costs "
                "O(n) total, giving O(n log n). Space is O(n) for the merge buffers."
            ),
            confidence="high",
        )
    return None


def _rule_divide_and_conquer_partition(f: dict):
    """Recursion + swap-in-loop, no merge step: Quick-Sort-style partitioning."""
    if f["has_recursion"] and f["recursion_branch_count"] >= 2 and f["has_swap_in_loop"]:
        return ComplexityEstimate(
            time="O(n log n) average, O(n^2) worst case",
            space="O(log n) average (recursion stack)",
            reasoning=(
                "Recursive function with an in-place swap pattern and 2 recursive "
                "calls - the classic partition-based divide-and-conquer shape (e.g. "
                "Quick Sort). Average case is O(n log n) with balanced partitions, "
                "but a poor pivot choice can degrade to O(n^2) in the worst case. "
                "Static analysis can't determine pivot quality, so both bounds are "
                "reported rather than picking one with false confidence."
            ),
            confidence="low",
        )
    return None


def _rule_tree_recursion(f: dict):
    """Recursion over a tree structure (self.left/self.right or root.left/right)."""
    if f["has_recursion"] and f["has_tree_pattern"]:
        return ComplexityEstimate(
            time="O(n)",
            space="O(h) where h is tree height (O(log n) balanced, O(n) worst case)",
            reasoning=(
                "Recursive function operating on a tree structure (left/right child "
                "access detected). Each node is visited once, giving O(n) time. "
                "Space is bounded by the recursion depth, i.e. the tree height - "
                "O(log n) for a balanced tree, degrading to O(n) for a skewed one."
            ),
            confidence="medium",
        )
    return None


def _rule_fast_slow_pointer(f: dict):
    """Floyd's cycle detection / linked list fast-slow traversal."""
    if f["has_fast_slow_pointer_pattern"]:
        return ComplexityEstimate(
            time="O(n)",
            space="O(1)",
            reasoning=(
                "Detected the fast/slow pointer pattern (fast advancing via "
                "double .next access). A single pass through the list with two "
                "pointers, no extra data structures - linear time, constant space."
            ),
            confidence="high",
        )
    return None


def _rule_sliding_window(f: dict):
    """Sliding window: each element enters and leaves the window at most once."""
    if f["has_sliding_window_pattern"]:
        return ComplexityEstimate(
            time="O(n)",
            space="O(1) or O(k) if tracking window contents in a set/dict",
            reasoning=(
                "Detected a sliding-window shape (a shrinking inner while-loop "
                "nested in an outer loop). Although this looks like two nested "
                "loops, each index only ever moves forward, so total work across "
                "both loops combined is bounded by O(n), not O(n^2)."
            ),
            confidence="high",
        )
    return None


def _rule_monotonic_stack(f: dict):
    """Monotonic stack: each element pushed and popped at most once."""
    if f["has_monotonic_stack_pattern"]:
        return ComplexityEstimate(
            time="O(n)",
            space="O(n)",
            reasoning=(
                "Detected a monotonic stack shape (conditional pop before push, "
                "driven by a comparison against the stack top). Although there is "
                "a loop with a nested while-pop, each element is pushed and popped "
                "at most once across the entire run, giving amortized O(n) time. "
                "Space is O(n) for the stack in the worst case (no pops needed)."
            ),
            confidence="high",
        )
    return None


def _rule_topological_sort(f: dict):
    """Kahn's algorithm: visits every node and edge once."""
    if f["has_topological_sort_pattern"]:
        return ComplexityEstimate(
            time="O(V + E)",
            space="O(V + E)",
            reasoning=(
                "Detected the indegree-array + queue (Kahn's algorithm) signature. "
                "Every vertex is enqueued/dequeued once and every edge is inspected "
                "once when decrementing indegrees, giving O(V + E) time. Space is "
                "O(V + E) for the adjacency list and indegree array."
            ),
            confidence="high",
        )
    return None


def _rule_bfs_dfs(f: dict):
    """Graph traversal via explicit queue/visited-set (BFS) or recursion (DFS)."""
    if (f["uses_queue"] or f["has_recursion"]) and f["uses_adjacency_structure"]:
        return ComplexityEstimate(
            time="O(V + E)",
            space="O(V)",
            reasoning=(
                "Detected graph traversal over an adjacency structure (queue-based "
                "or recursive). Every vertex is visited once and every edge is "
                "inspected once, giving O(V + E) time. Space is O(V) for the "
                "visited-tracking structure."
            ),
            confidence="medium",
        )
    return None


def _rule_bfs_no_explicit_adjacency_name(f: dict):
    """BFS via queue + visited set, even if adjacency wasn't caught by naming heuristic."""
    if f["uses_queue"] and f["has_visited_like_var"]:
        return ComplexityEstimate(
            time="O(V + E)",
            space="O(V)",
            reasoning=(
                "Detected a queue-driven traversal with visited-tracking - the BFS "
                "shape. Every vertex is processed once and every outgoing edge is "
                "inspected once, giving O(V + E) time and O(V) space."
            ),
            confidence="medium",
        )
    return None


def _rule_two_pointer_same_direction(f: dict):
    """Same-direction two pointer (in-place compaction): single pass, O(1) extra space."""
    if f["has_same_direction_pointer_pattern"] or f["has_two_pointer_pattern"] or f["has_parallel_two_pointer_pattern"]:
        return ComplexityEstimate(
            time="O(n)",
            space="O(1)",
            reasoning=(
                "Detected a two-pointer pattern. Both pointers only ever move "
                "forward (or toward each other), so the total number of steps "
                "across the whole run is bounded by O(n), with no extra data "
                "structure needed beyond the two index variables."
            ),
            confidence="high",
        )
    return None


def _rule_distribution_sort(f: dict):
    """Counting/Radix/Bucket sort: no comparisons, no recursion, uses auxiliary
    list-of-lists or frequency array."""
    if (
        not f["has_recursion"]
        and not f["has_swap_in_loop"]
        and f["max_loop_depth"] <= 2
        and (f["uses_dict"] or f["array_dimensionality"] >= 1)
        and not f["uses_sorted_or_sort_call"]
        and f["num_loops"] >= 2
    ):
        return ComplexityEstimate(
            time="O(n + k)",
            space="O(n + k)",
            reasoning=(
                "No comparisons or recursion detected, but multiple passes building "
                "a frequency/bucket structure - characteristic of a non-comparison "
                "sort (counting/radix/bucket). Time and space scale with n (input "
                "size) plus k (the value range or number of buckets)."
            ),
            confidence="low",
        )
    return None


def _rule_2d_dp_iterative(f: dict):
    """Iterative 2D DP table fill, no recursion."""
    if not f["has_recursion"] and f["array_dimensionality"] >= 2 and f["max_loop_depth"] >= 2:
        return ComplexityEstimate(
            time="O(n * m)",
            space="O(n * m)",
            reasoning=(
                "Iterative nested loops filling a 2D table (dp[i][j]-style), no "
                "recursion. Each cell is computed once from previously computed "
                "cells, giving time and space proportional to the table size."
            ),
            confidence="high",
        )
    return None


def _rule_nested_loops_swap(f: dict):
    """O(n^2) comparison-based in-place sort: nested loops + swap, no recursion."""
    if not f["has_recursion"] and f["max_loop_depth"] == 2 and f["has_swap_in_loop"]:
        return ComplexityEstimate(
            time="O(n^2)",
            space="O(1)",
            reasoning=(
                "Two nested loops with an in-place element swap, no recursion - the "
                "classic O(n^2) comparison-sort shape (bubble/selection/comb/cycle "
                "sort style). No extra space beyond a few scalar variables."
            ),
            confidence="high",
        )
    return None


def _rule_nested_loops_generic(f: dict):
    """Generic nested-loop fallback based purely on max_loop_depth."""
    depth = int(f["max_loop_depth"])
    if depth >= 2:
        power = "n^2" if depth == 2 else f"n^{depth}"
        return ComplexityEstimate(
            time=f"O({power})",
            space="O(1)" if not (f["uses_dict"] or f["uses_set"] or f["array_dimensionality"] >= 1) else "O(n)",
            reasoning=(
                f"Loops nested {depth} levels deep with no recursion and no "
                "faster-pattern signature (no sliding window, no monotonic stack, "
                "etc.) matched. Falling back to the direct nested-loop bound."
            ),
            confidence="low",
        )
    return None


def _rule_single_loop(f: dict):
    """A single loop, no recursion, no faster/slower pattern matched: O(n)."""
    if not f["has_recursion"] and f["max_loop_depth"] == 1:
        extra_space = f["uses_dict"] or f["uses_set"] or f["array_dimensionality"] >= 1
        return ComplexityEstimate(
            time="O(n)",
            space="O(n)" if extra_space else "O(1)",
            reasoning=(
                "A single loop over the input, no recursion. Linear time. Space is "
                "O(n) if an auxiliary set/dict/array is being built, otherwise O(1)."
            ),
            confidence="medium",
        )
    return None


def _rule_recursion_generic_fallback(f: dict):
    """Any remaining recursive case not caught by a more specific rule above."""
    if f["has_recursion"]:
        return ComplexityEstimate(
            time="O(n) to O(2^n) (recursion shape not fully resolved by static rules)",
            space="O(n)",
            reasoning=(
                "Recursion detected, but the specific shape didn't match any of the "
                "more specific rules (memoization, tree, divide-and-conquer, "
                "backtracking). Static feature analysis alone can't always pin down "
                "the exact bound here - manual review is recommended for a precise "
                "answer. Space is at least O(n) for the recursion call stack."
            ),
            confidence="low",
        )
    return None


def _rule_constant(f: dict):
    """No loop, no recursion: O(1)."""
    if not f["has_recursion"] and f["max_loop_depth"] == 0:
        return ComplexityEstimate(
            time="O(1)",
            space="O(1)",
            reasoning="No loops and no recursion detected - constant time and space.",
            confidence="medium",
        )
    return None


# Ordered from most specific/confident to most general. First match wins.
_RULES = [
    _rule_binary_search,
    _rule_union_find,
    _rule_topological_sort,
    _rule_fast_slow_pointer,
    _rule_sliding_window,
    _rule_monotonic_stack,
    _rule_backtracking,
    _rule_divide_and_conquer_merge,
    _rule_divide_and_conquer_partition,
    _rule_tree_recursion,
    _rule_recursive_memoized,
    _rule_recursive_unmemoized_branching,
    _rule_bfs_dfs,
    _rule_bfs_no_explicit_adjacency_name,
    _rule_two_pointer_same_direction,
    _rule_2d_dp_iterative,
    _rule_distribution_sort,
    _rule_nested_loops_swap,
    _rule_nested_loops_generic,
    _rule_single_loop,
    _rule_recursion_generic_fallback,
    _rule_constant,
]


def estimate_complexity_from_features(features: dict) -> Dict[str, str]:
    """Apply the rule list to an already-extracted feature dict."""
    for rule in _RULES:
        result = rule(features)
        if result is not None:
            return result.to_dict()

    # Should be unreachable given _rule_constant as the final catch-all,
    # but kept as a hard safety net.
    return ComplexityEstimate(
        time="Unknown",
        space="Unknown",
        reasoning="No rule matched this feature combination - this shouldn't happen; please report this case.",
        confidence="low",
    ).to_dict()


def estimate_complexity(code: str) -> Dict[str, str]:
    """Parse `code`, extract features, and return a complexity estimate dict:
    {"time_complexity": ..., "space_complexity": ..., "reasoning": ..., "confidence": ...}
    """
    features = extract_features(code)
    return estimate_complexity_from_features(features)


if __name__ == "__main__":
    import json

    samples = {
        "binary_search": """
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
        "bubble_sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
        "fibonacci_naive": """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
""",
        "fibonacci_memo": """
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
""",
        "lcs": """
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
""",
        "bfs": """
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
""",
        "union_find": """
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        parent[root_x] = root_y
""",
    }

    for name, code in samples.items():
        print(f"\n=== {name} ===")
        print(json.dumps(estimate_complexity(code), indent=2))