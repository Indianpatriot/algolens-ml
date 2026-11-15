import unittest
import model
from feature_extractor import extract_features, extract_top_level_parameters
from complexity_estimator import estimate_complexity

class TestCoinChangeBackend(unittest.TestCase):
    def test_coin_change_min_coins(self):
        code = '''def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount]

coins = [1, 2, 5]
amount = 11
'''
        f = extract_features(code)
        self.assertEqual(f["has_coin_change_min_pattern"], 1.0)
        self.assertEqual(f["has_coin_change_ways_pattern"], 0.0)
        self.assertEqual(f["has_knapsack_01_pattern"], 0.0)
        self.assertEqual(f["has_count_sort_rebuild"], 0.0)

        algo, conf = model.predict_algorithm(code, f)
        self.assertIn("Dynamic Programming", algo)
        self.assertIn("Coin Change", algo)
        self.assertNotIn("Knapsack", algo)
        self.assertNotIn("Counting Sort", algo)
        self.assertGreaterEqual(conf, 0.95)

        params = extract_top_level_parameters(code)
        self.assertEqual(params.get("coins"), [1, 2, 5])
        self.assertEqual(params.get("amount"), 11)

        comp = estimate_complexity(code, algorithm="Coin Change")
        self.assertEqual(comp["time_complexity"], "O(amount * len(coins))")
        self.assertEqual(comp["space_complexity"], "O(amount)")
        self.assertIn("[1, 2, 5]", comp["reasoning"])

    def test_coin_change_number_of_ways(self):
        code = '''def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]

    return dp[amount]

# change(5, [1, 2, 5])
'''
        f = extract_features(code)
        self.assertEqual(f["has_coin_change_ways_pattern"], 1.0)
        self.assertEqual(f["has_coin_change_min_pattern"], 0.0)
        self.assertEqual(f["has_knapsack_01_pattern"], 0.0)
        self.assertEqual(f["has_count_sort_rebuild"], 0.0)

        algo, conf = model.predict_algorithm(code, f)
        self.assertIn("Dynamic Programming", algo)
        self.assertIn("Coin Change", algo)
        self.assertIn("Ways", algo)
        self.assertNotIn("Counting Sort", algo)
        self.assertNotIn("Knapsack", algo)
        self.assertGreaterEqual(conf, 0.95)

        params = extract_top_level_parameters(code)
        self.assertEqual(params.get("coins"), [1, 2, 5])
        self.assertEqual(params.get("amount"), 5)

        comp = estimate_complexity(code, algorithm="Coin Change (Number of Ways)")
        self.assertEqual(comp["time_complexity"], "O(amount * len(coins))")
        self.assertEqual(comp["space_complexity"], "O(amount)")
        self.assertIn("combinations", comp["reasoning"])

    def test_knapsack_01(self):
        code = '''def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# knapsack([1, 3, 4, 5], [10, 40, 50, 70], 7)
'''
        f = extract_features(code)
        self.assertEqual(f["has_knapsack_01_pattern"], 1.0)
        self.assertEqual(f["has_coin_change_min_pattern"], 0.0)
        self.assertEqual(f["has_coin_change_ways_pattern"], 0.0)
        self.assertEqual(f["has_2d_dp_table"], 1.0)

        algo, conf = model.predict_algorithm(code, f)
        self.assertIn("Dynamic Programming", algo)
        self.assertIn("Knapsack", algo)
        self.assertNotIn("Coin Change", algo)
        self.assertGreaterEqual(conf, 0.95)

        params = extract_top_level_parameters(code)
        self.assertEqual(params.get("weights"), [1, 3, 4, 5])
        self.assertEqual(params.get("values"), [10, 40, 50, 70])
        self.assertEqual(params.get("capacity"), 7)

        comp = estimate_complexity(code, algorithm="0/1 Knapsack")
        self.assertEqual(comp["time_complexity"], "O(n * W)")
        self.assertEqual(comp["space_complexity"], "O(n * W)")
        self.assertIn("inclusion-exclusion", comp["reasoning"])

if __name__ == "__main__":
    unittest.main()
