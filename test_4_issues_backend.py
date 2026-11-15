import json
import urllib.request
from feature_extractor import extract_features
from model import predict_algorithm
from complexity_estimator import estimate_complexity_from_features, CANONICAL_ALGORITHM_COMPLEXITY

def test_word_break():
    print("Testing Issue 1: Word Break classification and complexity...")
    code = """def wordBreak(s, wordDict):
    dp = [False] * (len(s) + 1)
    dp[0] = True
    word_set = set(wordDict)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
"""
    features = extract_features(code)
    # Check discriminative features
    assert features["has_word_break_pattern"] == 1.0, "Expected has_word_break_pattern == 1.0"
    assert features["has_frequency_array"] == 0.0, "Expected has_frequency_array == 0.0 (not counting sort)"
    assert features["has_string_slicing_subscript"] == 1.0, "Expected has_string_slicing_subscript == 1.0"
    assert features["has_earlier_dp_dependency"] == 1.0, "Expected has_earlier_dp_dependency == 1.0"

    pred, conf = predict_algorithm(code, features)
    assert pred == "Dynamic Programming", f"Expected 'Dynamic Programming', got '{pred}'"
    assert conf >= 0.85, f"Expected confidence >= 0.85, got {conf}"

    cx = estimate_complexity_from_features(features, algorithm=pred, class_confidence=conf)
    assert cx["time_complexity"] == "O(n^2)", f"Expected O(n^2), got {cx['time_complexity']}"
    assert cx["space_complexity"] == "O(n)", f"Expected O(n), got {cx['space_complexity']}"
    assert cx["confidence"] == "high", f"Expected high confidence, got {cx['confidence']}"
    print("  ✓ Word Break is classified as Dynamic Programming (0.99) with O(n^2) / O(n) High Confidence complexity")

def test_counting_sort():
    print("Testing Issue 1: Counting Sort classification...")
    code = """def counting_sort(arr):
    if not arr:
        return []
    max_val = max(arr)
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    res = []
    for val, freq in enumerate(count):
        res.extend([val] * freq)
    return res
"""
    features = extract_features(code)
    assert features["has_frequency_array"] == 1.0, "Expected has_frequency_array == 1.0"
    assert features["has_element_indexed_count"] == 1.0, "Expected has_element_indexed_count == 1.0"
    assert features["has_word_break_pattern"] == 0.0, "Expected has_word_break_pattern == 0.0"

    pred, conf = predict_algorithm(code, features)
    assert pred == "Counting Sort", f"Expected 'Counting Sort', got '{pred}'"
    assert conf >= 0.85, f"Expected confidence >= 0.85, got {conf}"

    cx = estimate_complexity_from_features(features, algorithm=pred, class_confidence=conf)
    assert cx["time_complexity"] == "O(n + k)", f"Expected O(n + k), got {cx['time_complexity']}"
    assert cx["space_complexity"] == "O(k)", f"Expected O(k), got {cx['space_complexity']}"
    print("  ✓ Counting Sort correctly identified with O(n + k) / O(k) High Confidence")

def test_level_order_traversal():
    print("Testing Issue 4: Level Order Traversal complexity...")
    code = """def levelOrder(root):
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.pop(0)
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
    return result
"""
    features = extract_features(code)
    assert features["has_tree_pattern"] == 1.0, "Expected has_tree_pattern == 1.0"
    assert features["has_queue_while_loop"] == 1.0, "Expected has_queue_while_loop == 1.0"

    pred, conf = predict_algorithm(code, features)
    assert "BFS" in pred or "Tree" in pred, f"Expected BFS or Tree, got {pred}"

    cx = estimate_complexity_from_features(features, algorithm=pred, class_confidence=conf)
    assert cx["time_complexity"] == "O(n)", f"Expected O(n) time, got {cx['time_complexity']}"
    assert "O(w)" in cx["space_complexity"] or "O(n)" in cx["space_complexity"], f"Expected O(w) or O(n) space, got {cx['space_complexity']}"
    assert cx["confidence"] == "high", f"Expected high confidence, got {cx['confidence']}"
    print("  ✓ Level Order Traversal has O(n) time, O(w) <= O(n) space, High confidence")

def test_graph_bfs():
    print("Testing Issue 4: Graph BFS complexity preservation...")
    code = """def bfs(graph, start):
    visited = set([start])
    queue = [start]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
"""
    features = extract_features(code)
    assert features["has_tree_pattern"] == 0.0, "Expected has_tree_pattern == 0.0"

    pred, conf = predict_algorithm(code, features)
    assert pred == "BFS", f"Expected BFS, got {pred}"

    cx = estimate_complexity_from_features(features, algorithm=pred, class_confidence=conf)
    assert cx["time_complexity"] == "O(V + E)", f"Expected O(V + E) time, got {cx['time_complexity']}"
    assert cx["space_complexity"] == "O(V)", f"Expected O(V) space, got {cx['space_complexity']}"
    assert cx["confidence"] == "high", f"Expected high confidence, got {cx['confidence']}"
    print("  ✓ Graph BFS preserved as O(V + E) time, O(V) space, High confidence")

def test_fastapi_endpoint():
    print("Testing FastAPI live endpoint...")
    code = """def wordBreak(s, wordDict):
    dp = [False] * (len(s) + 1)
    dp[0] = True
    word_set = set(wordDict)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
"""
    req = urllib.request.Request(
        "http://127.0.0.1:8000/analyze",
        data=json.dumps({"code": code}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert data["algorithm"] == "Dynamic Programming"
        assert data["confidence"] >= 0.85
        assert data["complexity"]["time_complexity"] == "O(n^2)"
        assert data["complexity"]["space_complexity"] == "O(n)"
        assert data["complexity"]["confidence"] == "high"
    print("  ✓ Live FastAPI endpoint returns Dynamic Programming, O(n^2), O(n), high confidence")

if __name__ == "__main__":
    test_word_break()
    test_counting_sort()
    test_level_order_traversal()
    test_graph_bfs()
    test_fastapi_endpoint()
    print("\nALL BACKEND REGRESSION TESTS PASSED!")

