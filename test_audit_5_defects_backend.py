import unittest
from model import predict_algorithm
from feature_extractor import extract_features, extract_top_level_parameters
from complexity_estimator import estimate_complexity


class TestAudit5DefectsBackend(unittest.TestCase):
    def test_ast_constant_extraction_on_python_314(self):
        code = """
nums = [-1, 2, 3]
word = "ABC"

def solve_n_queens(n):
    pass

def coin_change(coins, amount):
    pass

solve_n_queens(4)
coin_change([1, 2, 5], 11)
"""
        params = extract_top_level_parameters(code)

        self.assertEqual(params["nums"], [-1, 2, 3])
        self.assertEqual(params["word"], "ABC")
        self.assertEqual(params["n"], 4)
        self.assertEqual(params["coins"], [1, 2, 5])
        self.assertEqual(params["amount"], 11)
        self.assertIsInstance(extract_features(code), dict)

    # -------------------------------------------------------------------------
    # DEFECT 1: Permutations Dynamic Input Extraction & Complexity
    # -------------------------------------------------------------------------
    def test_permutations_numbers(self):
        code = """
def permute(nums):
    res = []
    def backtrack(first=0):
        if first == len(nums):
            res.append(nums[:])
            return
        for i in range(first, len(nums)):
            nums[first], nums[i] = nums[i], nums[first]
            backtrack(first + 1)
            nums[first], nums[i] = nums[i], nums[first]
    backtrack()
    return res

# permute([9,8,7])
"""
        params = extract_top_level_parameters(code)
        self.assertIn("nums", params)
        self.assertEqual(params["nums"], [9, 8, 7])

        algo, conf = predict_algorithm(code)
        self.assertEqual(algo, "Permutations")
        self.assertGreaterEqual(conf, 0.95)

        comp = estimate_complexity(code, algo, conf)
        self.assertEqual(comp["time_complexity"], "O(n * n!)")
        self.assertEqual(comp["space_complexity"], "O(n)")
        self.assertIn("3! = 6 permutations", comp["reasoning"])
        self.assertIn("[9, 8, 7]", comp["reasoning"])

    def test_permutations_strings(self):
        code = """
def permute(nums):
    pass
permute(['a', 'b', 'c', 'd'])
"""
        params = extract_top_level_parameters(code)
        self.assertIn("nums", params)
        self.assertEqual(params["nums"], ["a", "b", "c", "d"])

        algo, conf = predict_algorithm(code)
        self.assertEqual(algo, "Permutations")

        comp = estimate_complexity(code, algo, conf)
        self.assertIn("4! = 24 permutations", comp["reasoning"])

    # -------------------------------------------------------------------------
    # DEFECT 2: Word Search Dynamic Board & Word Extraction
    # -------------------------------------------------------------------------
    def test_word_search_custom_grid(self):
        code = """
board = [["A", "B"], ["C", "D"]]
word = "ABDC"

def exist(board, word):
    pass
"""
        params = extract_top_level_parameters(code)
        self.assertEqual(params["board"], [["A", "B"], ["C", "D"]])
        self.assertEqual(params["word"], "ABDC")

        algo, conf = predict_algorithm(code)
        self.assertEqual(algo, "Word Search")
        self.assertGreaterEqual(conf, 0.95)

        comp = estimate_complexity(code, algo, conf)
        self.assertEqual(comp["time_complexity"], "O(R * C * 4^L)")
        self.assertEqual(comp["space_complexity"], "O(L)")
        self.assertIn("2×2 board", comp["reasoning"])
        self.assertIn('"ABDC"', comp["reasoning"])
        self.assertIn("length 4", comp["reasoning"])

    def test_word_search_single_row(self):
        code = """
board = [["A", "B", "C", "D"]]
word = "BC"
def exist(board, word):
    pass
"""
        params = extract_top_level_parameters(code)
        self.assertEqual(params["board"], [["A", "B", "C", "D"]])
        self.assertEqual(params["word"], "BC")

        algo, conf = predict_algorithm(code)
        self.assertEqual(algo, "Word Search")

        comp = estimate_complexity(code, algo, conf)
        self.assertIn("1×4 board", comp["reasoning"])
        self.assertIn('"BC"', comp["reasoning"])

    def test_word_search_invocation_call(self):
        code = """
def exist(board, word):
    pass
exist([["X", "Y"], ["Z", "W"]], "XW")
"""
        params = extract_top_level_parameters(code)
        self.assertEqual(params["board"], [["X", "Y"], ["Z", "W"]])
        self.assertEqual(params["word"], "XW")

        algo, conf = predict_algorithm(code)
        self.assertEqual(algo, "Word Search")

    # -------------------------------------------------------------------------
    # DEFECT 3: N-Queens Dynamic N Extraction (n=8)
    # -------------------------------------------------------------------------
    def test_n_queens_eight(self):
        code = """
def solve_n_queens(n):
    cols, d1, d2 = set(), set(), set()
    board = [-1] * n
    res = []
    def backtrack(r):
        if r == n:
            res.append(board[:])
            return
        for c in range(n):
            if c in cols or (r - c) in d1 or (r + c) in d2:
                continue
            cols.add(c); d1.add(r - c); d2.add(r + c)
            board[r] = c
            backtrack(r + 1)
            cols.remove(c); d1.remove(r - c); d2.remove(r + c)
    backtrack(0)
    return res

n = 8
"""
        params = extract_top_level_parameters(code)
        self.assertEqual(params.get("n"), 8)

        algo, conf = predict_algorithm(code)
        self.assertEqual(algo, "N-Queens")
        self.assertGreaterEqual(conf, 0.95)

        comp = estimate_complexity(code, algo, conf)
        self.assertEqual(comp["time_complexity"], "O(n!)")
        self.assertEqual(comp["space_complexity"], "O(n)")
        self.assertIn("n = 8 (8×8 chessboard)", comp["reasoning"])
        self.assertIn("Total valid solutions: 92", comp["reasoning"])

    # -------------------------------------------------------------------------
    # DEFECT 4: Coin Change vs 0/1 Knapsack
    # -------------------------------------------------------------------------
    def test_coin_change_vs_knapsack(self):
        code_min = """
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount]
"""
        algo_min, conf_min = predict_algorithm(code_min)
        self.assertEqual(algo_min, "Dynamic Programming (Coin Change)")
        self.assertGreaterEqual(conf_min, 0.95)

        code_ways = """
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    return dp[amount]
"""
        algo_ways, conf_ways = predict_algorithm(code_ways)
        self.assertEqual(algo_ways, "Dynamic Programming (Coin Change - Number of Ways)")
        self.assertGreaterEqual(conf_ways, 0.95)

        code_knap = """
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w-weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]
"""
        algo_knap, conf_knap = predict_algorithm(code_knap)
        self.assertEqual(algo_knap, "Dynamic Programming (0/1 Knapsack)")
        self.assertGreaterEqual(conf_knap, 0.95)

    # -------------------------------------------------------------------------
    # DEFECT 5: Level Order Traversal vs Invert Binary Tree
    # -------------------------------------------------------------------------
    def test_level_order_vs_invert_tree(self):
        code_level_order = """
from collections import deque
def level_order(root):
    if not root:
        return []
    res = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res
"""
        algo_lo, conf_lo = predict_algorithm(code_level_order)
        self.assertEqual(algo_lo, "Level Order Traversal")
        self.assertGreaterEqual(conf_lo, 0.95)

        comp_lo = estimate_complexity(code_level_order, algo_lo, conf_lo)
        self.assertEqual(comp_lo["time_complexity"], "O(n)")
        self.assertIn("O(w) <= O(n)", comp_lo["space_complexity"])
        self.assertIn("Breadth-first search level-order traversal on a binary tree", comp_lo["reasoning"])

        code_invert = """
def invert_tree(root):
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root
"""
        algo_inv, conf_inv = predict_algorithm(code_invert)
        self.assertEqual(algo_inv, "Invert Binary Tree")
        self.assertGreaterEqual(conf_inv, 0.95)


if __name__ == "__main__":
    unittest.main()
