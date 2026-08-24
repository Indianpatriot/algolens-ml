from feature_extractor import extract_features
import json

SAMPLES = {
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
    "fibonacci_memo": """
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
""",
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

"n_queens_backtracking": """
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
    "floyd_cycle_detection": """
def has_cycle(head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
""",
    "sliding_window_max_sum": """
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
""",
    "next_greater_element_monotonic_stack": """
def next_greater_elements(nums):
    result = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
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
    "quick_sort": """
def quick_sort(arr, start=0, end=None):
    if end is None:
        end = len(arr) - 1
    if start < end:
        pivot = arr[end]
        i = start - 1
        for j in range(start, end):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[end] = arr[end], arr[i + 1]
        pivot_idx = i + 1
        quick_sort(arr, start, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, end)
    return arr
""",
    "merge_sort": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
    "insertion_sort": """
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
""",
    "shell_sort": """
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            tmp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > tmp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = tmp
        gap //= 2
    return arr
""",
    "heap_sort": """
def heap_sort(arr):
    n = len(arr)
    def heapify(heap_size, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < heap_size and arr[l] > arr[largest]:
            largest = l
        if r < heap_size and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(heap_size, largest)
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(i, 0)
    return arr
""",
    "counting_sort": """
def counting_sort(arr):
    if not arr:
        return arr
    minv, maxv = min(arr), max(arr)
    span = maxv - minv + 1
    count = [0] * span
    for num in arr:
        count[num - minv] += 1
    for i in range(1, span):
        count[i] += count[i - 1]
    res = [0] * len(arr)
    for num in reversed(arr):
        count[num - minv] -= 1
        res[count[num - minv]] = num
    return res
""",
    "radix_sort": """
def radix_sort(arr):
    if not arr:
        return arr
    maxv = max(arr)
    exp = 1
    while maxv // exp > 0:
        count = [0] * 10
        for x in arr:
            count[(x // exp) % 10] += 1
        exp *= 10
    return arr
""",
}

if __name__ == "__main__":
    for name, code in SAMPLES.items():
        print(f"\n=== {name} ===")
        try:
            features = extract_features(code)
            nonzero = {k: v for k, v in features.items() if v != 0.0}
            print(json.dumps(nonzero, indent=2))
        except SyntaxError as e:
            print(f"SYNTAX ERROR: {e}")
            