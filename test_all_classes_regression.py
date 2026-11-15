import urllib.request
import json

BASE_URL = "http://127.0.0.1:8000/analyze"

def query_api(code: str):
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps({"code": code}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        assert res.status == 200, f"API error {res.status}"
        return json.loads(res.read().decode("utf-8"))

TEST_CASES = [
    ("Binary Search", """
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""", ["Binary Search"]),

    ("Reverse Linked List", """
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
""", ["Linked List", "Linked List Operations", "Reverse Linked List"]),

    ("Floyd Cycle Detection", """
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
""", ["Linked List", "Linked List Operations", "Floyd Cycle Detection", "Two Pointer"]),

    ("Tree Max Depth", """
def max_depth(root):
    if not root:
        return 0
    left = max_depth(root.left)
    right = max_depth(root.right)
    return max(left, right) + 1
""", ["Tree", "Binary Tree", "DFS", "Tree Operations"]),

    ("Monotonic Stack", """
def next_greater_element(nums):
    res = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            res[idx] = nums[i]
        stack.append(i)
    return res
""", ["Monotonic Stack", "Stack"]),

    ("Sliding Window", """
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
""", ["Sliding Window"]),

    ("Union-Find", """
parent = {}
def find(i):
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j
""", ["Union-Find"]),

    ("Dynamic Programming - LCS", """
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
""", ["Dynamic Programming"]),

    ("Topological Sort Kahn", """
from collections import deque
def topo_sort(num_nodes, edges):
    in_degree = [0] * num_nodes
    adj = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    q = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    return order
""", ["Topological Sort", "Topological Sort (Kahn's)", "Distribution Sort"]),

    ("Heap Sort", """
def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr
""", ["Heap Sort", "Heap/Complex Structural Sort"]),

    ("Counting Sort", """
def counting_sort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    idx = 0
    for i in range(len(count)):
        while count[i] > 0:
            arr[idx] = i
            idx += 1
            count[i] -= 1
    return arr
""", ["Counting Sort"]),

    ("Radix Sort", """
def radix_sort(arr):
    max1 = max(arr)
    exp = 1
    while max1 // exp > 0:
        count = [0] * 10
        output = [0] * len(arr)
        for i in range(len(arr)):
            index = (arr[i] // exp) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for i in range(len(arr) - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
        for i in range(len(arr)):
            arr[i] = output[i]
        exp *= 10
    return arr
""", ["Radix Sort"]),
]

def run_tests():
    passed = 0
    for name, code, expected_algs in TEST_CASES:
        res = query_api(code)
        algo = res.get("algorithm")
        conf = res.get("confidence")
        match = any(exp.lower() in algo.lower() or algo.lower() in exp.lower() for exp in expected_algs)
        status = "PASS" if match else "FAIL"
        print(f"[{status}] {name}: predicted='{algo}' (conf={conf:.2f}), expected one of {expected_algs}")
        assert match, f"Expected {expected_algs}, but got {algo}"
        passed += 1
    print(f"\nAll {passed} non-affected algorithm class tests PASSED with zero regressions!")

if __name__ == "__main__":
    run_tests()
