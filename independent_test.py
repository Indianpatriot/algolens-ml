"""
Independent generalization check.

These samples were hand-written in Phase 2, BEFORE the dataset generator
(common.py / generate_dataset.py / templates_*.py) existed. They were never
part of the template-rendering pipeline, so this is a genuine out-of-
distribution test - a much more honest check of generalization than
cross-validation on the template-generated dataset alone.

Run from project root: python independent_test.py
"""
import joblib
import pandas as pd
from pathlib import Path
from feature_extractor import extract_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_ORDER_PATH = BASE_DIR / "feature_order.pkl"

# Only samples whose algorithm maps cleanly onto one of the 16 trained
# categories are included with an expected label. Samples covering
# algorithms NOT in the current taxonomy (Dijkstra, bit manipulation,
# generic tree traversal, prefix sum, topological sort, matrix transpose,
# modular exponentiation) are included separately, unlabeled, just to see
# what the model does when asked about something genuinely out of scope.

LABELED_SAMPLES = {
    "binary_search": ("""
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
""", "Binary Search"),

    "bubble_sort": ("""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""", "In-Place Nested-Loop Swap Sort"),

    "bfs": ("""
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
""", "BFS"),

    "fibonacci_memo": ("""
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
""", "Dynamic Programming"),

    "lcs": ("""
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
""", "Dynamic Programming"),

    "n_queens_backtracking": ("""
def solve_n_queens(n):
    results = []
    path = []

    def is_valid(path, row, col):
        for r, c in enumerate(path):
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True

    def backtrack(row):
        if row == n:
            results.append(path[:])
            return
        for col in range(n):
            if is_valid(path, row, col):
                path.append(col)
                backtrack(row + 1)
                path.pop()

    backtrack(0)
    return results
""", "Backtracking"),

    "union_find": ("""
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        parent[root_x] = root_y
""", "Union-Find"),

    "floyd_cycle_detection": ("""
def has_cycle(head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
""", "Two Pointer"),

    "sliding_window_max_sum": ("""
def max_sum_subarray(nums, k):
    window_sum = 0
    start = 0
    max_sum = 0
    for end in range(len(nums)):
        window_sum += nums[end]
        while end - start + 1 > k:
            window_sum -= nums[start]
            start += 1
        max_sum = max(max_sum, window_sum)
    return max_sum
""", "Sliding Window"),

    "next_greater_element_monotonic_stack": ("""
def next_greater_elements(nums):
    result = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
""", "Monotonic Stack"),
}

# Out-of-scope samples: algorithms not covered by the current 16-class
# taxonomy. Included to see what the model predicts when asked about
# something it was never trained to recognize - a low-confidence or
# clearly-wrong prediction here is EXPECTED and fine; it's not a failure,
# it's a scope boundary.
UNLABELED_SAMPLES = {
    "dijkstra": """
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances
""",
    "single_number_xor": """
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
""",
    "reverse_linked_list": """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
""",
    "tree_max_depth": """
def max_depth(root):
    if root is None:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return max(left_depth, right_depth) + 1
""",
    "prefix_sum": """
def prefix_sum(nums):
    n = len(nums)
    prefix = [0] * n
    prefix[0] = nums[0]
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + nums[i]
    return prefix
""",
    "topological_sort_kahns": """
from collections import deque

def topo_sort(num_nodes, edges):
    indegree = [0] * num_nodes
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    queue = deque([i for i in range(num_nodes) if indegree[i] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return order
""",
    "matrix_transpose": """
def transpose(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix
""",
    "modular_exponentiation": """
def mod_pow(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod
    return result
""",
}


def main():
    clf = joblib.load(MODEL_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)

    print("=" * 70)
    print("IN-SCOPE SAMPLES (expected label is one of the 16 trained classes)")
    print("=" * 70)
    correct = 0
    for name, (code, expected) in LABELED_SAMPLES.items():
        features = extract_features(code)
        row = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        pred = clf.predict(row)[0]
        proba = clf.predict_proba(row).max()
        is_correct = pred == expected
        correct += is_correct
        status = "OK  " if is_correct else "MISS"
        print(f"[{status}] {name:40s} expected={expected:35s} predicted={pred:35s} conf={proba:.2f}")

    print(f"\nIn-scope accuracy: {correct}/{len(LABELED_SAMPLES)} = {correct/len(LABELED_SAMPLES):.2%}")

    print("\n" + "=" * 70)
    print("OUT-OF-SCOPE SAMPLES (no trained category matches - for reference only)")
    print("=" * 70)
    for name, code in UNLABELED_SAMPLES.items():
        features = extract_features(code)
        row = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        pred = clf.predict(row)[0]
        proba = clf.predict_proba(row).max()
        print(f"        {name:40s} predicted={pred:35s} conf={proba:.2f}")


if __name__ == "__main__":
    main()