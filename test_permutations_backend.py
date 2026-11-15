import unittest
from feature_extractor import extract_top_level_parameters, extract_features
from complexity_estimator import estimate_complexity

class TestPermutationsBackend(unittest.TestCase):
    def test_extract_commented_permute_call(self):
        code = '''
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
'''
        params = extract_top_level_parameters(code)
        self.assertIn("nums", params)
        self.assertEqual(params["nums"], [9, 8, 7])

        features = extract_features(code)
        self.assertEqual(features.get("detected_nums_len"), 3.0)

        complexity = estimate_complexity(code, algorithm="Permutations")
        self.assertEqual(complexity["time_complexity"], "O(n * n!)")
        self.assertIn("3! = 6", complexity["reasoning"])
        self.assertIn("[9, 8, 7]", complexity["reasoning"])

    def test_extract_permute_strings(self):
        code = '''
def permute(nums):
    res = []
    backtrack(0)
    return res

permute(['a', 'b', 'c'])
'''
        params = extract_top_level_parameters(code)
        self.assertIn("nums", params)
        self.assertEqual(params["nums"], ["a", "b", "c"])

        features = extract_features(code)
        self.assertEqual(features.get("detected_nums_len"), 3.0)

        complexity = estimate_complexity(code, algorithm="Permutations")
        self.assertEqual(complexity["time_complexity"], "O(n * n!)")
        self.assertIn("['a', 'b', 'c']", complexity["reasoning"])

    def test_extract_permute_duplicates(self):
        code = '''
def permute(nums):
    pass
nums = [5, 5, 6]
permute(nums)
'''
        params = extract_top_level_parameters(code)
        self.assertIn("nums", params)
        self.assertEqual(params["nums"], [5, 5, 6])

if __name__ == "__main__":
    unittest.main()
