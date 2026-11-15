import urllib.request
import json
import sys

URL = "http://127.0.0.1:8000/analyze"

def analyze_code(code: str) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps({"code": code}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_backend_9_issues():
    passed = 0
    total = 0

    def assert_true(cond, msg):
        nonlocal passed, total
        total += 1
        if not cond:
            print(f"[FAIL] {msg}")
            sys.exit(1)
        passed += 1
        print(f"[PASS] {msg}")

    print("===============================================================")
    print("BACKEND REGRESSION TEST SUITE: 9 ISSUES VERIFICATION")
    print("===============================================================")

    # 1. Heap Sort
    print("\n--- 1. Heap Sort Backend ---")
    heap_code = """
def heap_sort(arr):
    n = len(arr)
    def heapify(n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(n, largest)

    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(i, 0)
    return arr
"""
    res1 = analyze_code(heap_code)
    assert_true(res1["algorithm"] == "Heap Sort", f"Heap sort algorithm is 'Heap Sort' (got '{res1['algorithm']}')")
    assert_true(res1["confidence"] >= 0.95, f"Heap sort confidence >= 0.95 (got {res1['confidence']})")
    assert_true(res1["complexity"]["confidence"] == "high", f"Heap sort complexity confidence is high (got '{res1['complexity']['confidence']}')")
    assert_true("O(n log n)" in res1["complexity"]["time_complexity"], f"Heap sort time O(n log n) (got '{res1['complexity']['time_complexity']}')")

    # 2. N-Queens
    print("\n--- 2. N-Queens Backend ---")
    nq_code = """
def solve_n_queens(n):
    board = [['.'] * n for _ in range(n)]
    res = []
    def backtrack(r):
        if r == n:
            res.append(["".join(row) for row in board])
            return
        for c in range(n):
            board[r][c] = 'Q'
            backtrack(r + 1)
            board[r][c] = '.'
    backtrack(0)
    return res
"""
    res2 = analyze_code(nq_code)
    assert_true(res2["algorithm"] == "N-Queens", f"N-Queens algorithm is 'N-Queens' (got '{res2['algorithm']}')")
    assert_true(res2["confidence"] >= 0.95, f"N-Queens confidence >= 0.95 (got {res2['confidence']})")
    assert_true(res2["complexity"]["confidence"] == "high", f"N-Queens complexity confidence is 'high' (got '{res2['complexity']['confidence']}')")
    assert_true("O(n!)" in res2["complexity"]["time_complexity"] or "O(k^n)" in res2["complexity"]["time_complexity"], f"N-Queens time is exponential/factorial (got '{res2['complexity']['time_complexity']}')")

    # 3. Permutations
    print("\n--- 3. Permutations Backend ---")
    perm_code = """
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                backtrack(path, used)
                path.pop()
                used[i] = False
    backtrack([], [False] * len(nums))
    return res
"""
    res3 = analyze_code(perm_code)
    assert_true(res3["algorithm"] == "Permutations", f"Permutations algorithm is 'Permutations' (got '{res3['algorithm']}')")
    assert_true(res3["confidence"] >= 0.95, f"Permutations confidence >= 0.95 (got {res3['confidence']})")
    assert_true(res3["complexity"]["confidence"] == "high", f"Permutations complexity confidence is 'high' (got '{res3['complexity']['confidence']}')")

    # 4. Word Search
    print("\n--- 4. Word Search Backend ---")
    ws_code = """
def exist(board, word):
    R, C = len(board), len(board[0])
    def dfs(r, c, k):
        if k == len(word): return True
        if r < 0 or r >= R or c < 0 or c >= C or board[r][c] != word[k]: return False
        tmp = board[r][c]
        board[r][c] = '#'
        found = dfs(r+1,c,k+1) or dfs(r-1,c,k+1) or dfs(r,c+1,k+1) or dfs(r,c-1,k+1)
        board[r][c] = tmp
        return found
    for r in range(R):
        for c in range(C):
            if dfs(r, c, 0): return True
    return False
"""
    res4 = analyze_code(ws_code)
    assert_true(res4["algorithm"] == "Word Search", f"Word Search algorithm is 'Word Search' (got '{res4['algorithm']}')")
    assert_true(res4["confidence"] >= 0.95, f"Word Search confidence >= 0.95 (got {res4['confidence']})")
    assert_true(res4["complexity"]["confidence"] == "high", f"Word Search complexity confidence is 'high' (got '{res4['complexity']['confidence']}')")

    # 5. Two Sum II
    print("\n--- 5. Two Sum II Backend ---")
    ts_code = """
def two_sum_ii(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        cur = numbers[left] + numbers[right]
        if cur == target:
            return [left + 1, right + 1]
        elif cur < target:
            left += 1
        else:
            right -= 1
    return []
"""
    res5 = analyze_code(ts_code)
    assert_true("Two Pointer" in res5["algorithm"], f"Two Sum II algorithm is 'Two Pointer' (got '{res5['algorithm']}')")
    assert_true(res5["confidence"] >= 0.90, f"Two Sum II confidence >= 0.90 (got {res5['confidence']})")
    assert_true("O(n)" in res5["complexity"]["time_complexity"], f"Two Sum II time is O(n) (got '{res5['complexity']['time_complexity']}')")

    # 6. Remove Duplicates
    print("\n--- 6. Remove Duplicates Backend ---")
    dup_code = """
def removeDuplicates(nums):
    if not nums: return 0
    write = 0
    for read in range(1, len(nums)):
        if nums[read] != nums[write]:
            write += 1
            nums[write] = nums[read]
    return write + 1
"""
    res6 = analyze_code(dup_code)
    assert_true("Two Pointer" in res6["algorithm"], f"Remove Duplicates algorithm is 'Two Pointer' (got '{res6['algorithm']}')")
    assert_true(res6["confidence"] >= 0.90, f"Remove Duplicates confidence >= 0.90 (got {res6['confidence']})")

    # 7. Coin Change
    print("\n--- 7. Coin Change Backend ---")
    cc_code = """
def coinChange(coins, amount):
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if i - c >= 0:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] <= amount else -1
"""
    res7 = analyze_code(cc_code)
    assert_true("Dynamic Programming" in res7["algorithm"], f"Coin Change algorithm is 'Dynamic Programming' (got '{res7['algorithm']}')")
    assert_true(res7["confidence"] >= 0.90, f"Coin Change confidence >= 0.90 (got {res7['confidence']})")

    # 8. LCS
    print("\n--- 8. LCS Backend ---")
    lcs_code = """
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
"""
    res8 = analyze_code(lcs_code)
    assert_true("Dynamic Programming" in res8["algorithm"], f"LCS algorithm is 'Dynamic Programming' (got '{res8['algorithm']}')")
    assert_true(res8["confidence"] >= 0.60, f"LCS confidence >= 0.60 (got {res8['confidence']})")

    # 9. 0/1 Knapsack
    print("\n--- 9. 0/1 Knapsack Backend ---")
    knap_code = """
def knapsack(W, wt, val, n):
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if wt[i-1] <= w:
                dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]
"""
    res9 = analyze_code(knap_code)
    assert_true("Dynamic Programming" in res9["algorithm"], f"0/1 Knapsack algorithm is 'Dynamic Programming' (got '{res9['algorithm']}')")
    assert_true(res9["confidence"] >= 0.90, f"0/1 Knapsack confidence >= 0.90 (got {res9['confidence']})")

    # Climbing Stairs verification
    print("\n--- Climbing Stairs Backend ---")
    cs_code = """
def climbStairs(n):
    if n <= 2: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
"""
    res10 = analyze_code(cs_code)
    assert_true("Dynamic Programming" in res10["algorithm"], f"Climbing Stairs algorithm is 'Dynamic Programming' (got '{res10['algorithm']}')")
    assert_true(res10["confidence"] >= 0.90, f"Climbing Stairs confidence >= 0.90 (got {res10['confidence']})")

    print("\n===============================================================")
    print(f"ALL BACKEND TESTS PASSED: {passed}/{total} assertions verified.")
    print("===============================================================")

if __name__ == "__main__":
    test_backend_9_issues()

