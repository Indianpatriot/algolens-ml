import json
import urllib.request
from feature_extractor import extract_features
from model import predict_algorithm
from complexity_estimator import estimate_complexity_from_features

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

# --- BUG 1: BFS ---
BFS_PRIMARY = """
def bfs(graph, start):
    visited = set([start]); queue = [start]; order = []
    while queue:
        node = queue.pop(0); order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor); queue.append(neighbor)
    return order
"""

BFS_DEQUE = """
from collections import deque
def bfs_deque(adj, src):
    seen = {src}
    q = deque([src])
    res = []
    while q:
        u = q.popleft()
        res.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return res
"""

BFS_RENAMED = """
def traverse(g, s):
    vis = set([s])
    q = [s]
    ans = []
    while q:
        curr = q.pop(0)
        ans.append(curr)
        for nxt in g[curr]:
            if nxt not in vis:
                vis.add(nxt)
                q.append(nxt)
    return ans
"""

BFS_GRID = """
def shortest_path_grid(grid, start_r, start_c):
    R, C = len(grid), len(grid[0])
    q = [(start_r, start_c, 0)]
    visited = {(start_r, start_c)}
    while q:
        r, c, dist = q.pop(0)
        if grid[r][c] == 'T':
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#' and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, dist + 1))
    return -1
"""

# --- BUG 2: BUCKET SORT ---
BUCKET_PRIMARY = """
def bucket_sort(arr):
    if not arr: return []
    n = len(arr)
    buckets = [[] for _ in range(n)]
    for num in arr:
        idx = int(n * num)
        if idx == n: idx = n - 1
        buckets[idx].append(num)
    for b in buckets:
        b.sort()
    result = []
    for b in buckets:
        result.extend(b)
    return result
"""

BUCKET_SCALED = """
def bucket_sort_integers(nums):
    if not nums: return []
    mn, mx = min(nums), max(nums)
    k = len(nums)
    slots = [[] for _ in range(k)]
    for val in nums:
        idx = int((val - mn) / (mx - mn + 1) * k)
        slots[idx].append(val)
    for s in slots:
        s.sort()
    out = []
    for s in slots:
        out.extend(s)
    return out
"""

BUCKET_INSERTION = """
def bucket_sort_insertion(arr):
    n = len(arr)
    buckets = [[] for _ in range(n)]
    for x in arr:
        i = int(n * x)
        buckets[i].append(x)
    for b in buckets:
        for p in range(1, len(b)):
            key = b[p]
            j = p - 1
            while j >= 0 and b[j] > key:
                b[j + 1] = b[j]
                j -= 1
            b[j + 1] = key
    ans = []
    for b in buckets:
        ans.extend(b)
    return ans
"""

BUCKET_GENERIC = """
def custom_bucket_sort(data):
    size = len(data)
    containers = [[] for _ in range(size)]
    for item in data:
        slot = int(item * size)
        if slot >= size: slot = size - 1
        containers[slot].append(item)
    for c in containers:
        c.sort()
    ordered = []
    for c in containers:
        ordered.extend(c)
    return ordered
"""

# --- BUG 3: QUICK SORT ---
QUICK_PRIMARY = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

QUICK_LOMUTO = """
def quicksort_lomuto(arr, low, high):
    if low < high:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        pi = i + 1
        quicksort_lomuto(arr, low, pi - 1)
        quicksort_lomuto(arr, pi + 1, high)
"""

QUICK_HOARE = """
def quicksort_hoare(arr, low, high):
    if low < high:
        pivot = arr[(low + high) // 2]
        i, j = low - 1, high + 1
        while True:
            i += 1
            while arr[i] < pivot: i += 1
            j -= 1
            while arr[j] > pivot: j -= 1
            if i >= j:
                p = j
                break
            arr[i], arr[j] = arr[j], arr[i]
        quicksort_hoare(arr, low, p)
        quicksort_hoare(arr, p + 1, high)
"""

# --- BUG 4: MERGE SORT ---
MERGE_PRIMARY = """
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
"""

MERGE_INDEX = """
def merge_sort_indices(arr):
    def sort(l, r):
        if l >= r: return
        m = (l + r) // 2
        sort(l, m)
        sort(m + 1, r)
        merge(l, m, r)
    def merge(l, m, r):
        temp = []
        i, j = l, m + 1
        while i <= m and j <= r:
            if arr[i] <= arr[j]:
                temp.append(arr[i])
                i += 1
            else:
                temp.append(arr[j])
                j += 1
        while i <= m:
            temp.append(arr[i]); i += 1
        while j <= r:
            temp.append(arr[j]); j += 1
        for k in range(len(temp)):
            arr[l + k] = temp[k]
    sort(0, len(arr) - 1)
    return arr
"""

MERGE_RENAMED = """
def ms(a):
    if len(a) <= 1: return a
    half = len(a) // 2
    L = ms(a[:half])
    R = ms(a[half:])
    merged = []
    p1 = p2 = 0
    while p1 < len(L) and p2 < len(R):
        if L[p1] <= R[p2]:
            merged.append(L[p1])
            p1 += 1
        else:
            merged.append(R[p2])
            p2 += 1
    merged.extend(L[p1:])
    merged.extend(R[p2:])
    return merged
"""

def test_bug1_bfs_api():
    cases = [
        ("BFS Primary", BFS_PRIMARY),
        ("BFS Deque", BFS_DEQUE),
        ("BFS Renamed", BFS_RENAMED),
        ("BFS Grid", BFS_GRID),
    ]
    for name, code in cases:
        data = query_api(code)
        print(f"[{name}] API response: algo={data.get('algorithm')}, conf={data.get('confidence')}")
        assert data.get("algorithm") == "BFS", f"{name} got {data.get('algorithm')}"
        assert data.get("confidence") >= 0.80, f"{name} confidence {data.get('confidence')} < 0.80"

def test_bug2_bucket_sort_api():
    cases = [
        ("Bucket Sort Primary", BUCKET_PRIMARY),
        ("Bucket Sort Scaled", BUCKET_SCALED),
        ("Bucket Sort Insertion", BUCKET_INSERTION),
        ("Bucket Sort Generic", BUCKET_GENERIC),
    ]
    for name, code in cases:
        data = query_api(code)
        print(f"[{name}] API response: algo={data.get('algorithm')}, conf={data.get('confidence')}")
        assert data.get("algorithm") == "Bucket Sort", f"{name} got {data.get('algorithm')}"
        assert data.get("confidence") >= 0.80, f"{name} confidence {data.get('confidence')} < 0.80"

def test_bug3_quicksort_complexity():
    cases = [
        ("Quick Sort Primary", QUICK_PRIMARY),
        ("Quick Sort Lomuto", QUICK_LOMUTO),
        ("Quick Sort Hoare", QUICK_HOARE),
    ]
    for name, code in cases:
        data = query_api(code)
        algo = data.get("algorithm")
        conf = data.get("confidence")
        comp = data.get("complexity", {})
        time_comp = comp.get("time_complexity")
        space_comp = comp.get("space_complexity")
        comp_conf = comp.get("confidence")
        print(f"[{name}] API response: algo={algo} ({conf:.2f}), time={time_comp}, space={space_comp}, comp_conf={comp_conf}")
        assert algo == "Quick Sort", f"{name} got {algo}"
        assert conf >= 0.80, f"{name} algo confidence {conf} < 0.80"
        assert comp_conf in ["high", "High"], f"{name} complexity confidence {comp_conf} is not High!"
        assert "n log n" in time_comp.lower() or "n^2" in time_comp.lower(), f"{name} time complexity unexpected: {time_comp}"
        assert "log n" in space_comp.lower(), f"{name} space complexity unexpected: {space_comp}"

def test_bug4_mergesort_api():
    cases = [
        ("Merge Sort Primary", MERGE_PRIMARY),
        ("Merge Sort Indices", MERGE_INDEX),
        ("Merge Sort Renamed", MERGE_RENAMED),
    ]
    for name, code in cases:
        data = query_api(code)
        algo = data.get("algorithm")
        conf = data.get("confidence")
        print(f"[{name}] API response: algo={algo} ({conf:.2f})")
        assert algo == "Merge Sort", f"{name} got {algo}"
        assert conf >= 0.80, f"{name} algo confidence {conf} < 0.80"

if __name__ == "__main__":
    print("Running Bug 1 tests...")
    test_bug1_bfs_api()
    print("\nRunning Bug 2 tests...")
    test_bug2_bucket_sort_api()
    print("\nRunning Bug 3 tests...")
    test_bug3_quicksort_complexity()
    print("\nRunning Bug 4 tests...")
    test_bug4_mergesort_api()
    print("\nAll API tests PASSED successfully!")
