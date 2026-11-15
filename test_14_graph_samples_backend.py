"""
test_14_graph_samples_backend.py

Regression test verifying all 14 graph algorithm samples from the user prompt:
Sample 3: BFS (deque, visited on pop) -> BFS, O(V + E) / O(V)
Sample 4: BFS (bounded for-loop) -> BFS, O(V + E) / O(V)
Sample 5: Iterative DFS (reversed) -> DFS (iterative), O(V + E) / O(V)
Sample 6: Iterative DFS (extend) -> DFS (iterative), O(V + E) / O(V)
Sample 7: Recursive DFS (nested explore) -> DFS, O(V + E) / O(V)
Sample 8: Dijkstra (lazy) -> Dijkstra's Algorithm, O((V + E) log V) / O(V)
Sample 9: Dijkstra (visited set) -> Dijkstra's Algorithm, O((V + E) log V) / O(V)
Sample 10: Bellman-Ford -> Bellman-Ford, O(V * E) / O(V)
Sample 11: 0-1 BFS -> 0-1 BFS, O(V + E) / O(V)
Sample 12: APSP (repeated Bellman-Ford) -> All-Pairs Shortest Path (repeated Bellman-Ford), O(V^2 * E) / O(V^2)
Sample 13: Topological Sort (Kahn) -> Topological Sort (Kahn), O(V + E) / O(V)
Sample 14: Topological Sort (DFS cycle) -> Topological Sort (DFS), O(V + E) / O(V)
Sample 15: DAG Shortest Path (DFS) -> DAG Shortest Path, O(V + E) / O(V)
Sample 16: DAG Shortest Path (Kahn) -> DAG Shortest Path, O(V + E) / O(V)
"""

from test_current_samples import samples
from feature_extractor import extract_features
from model import predict_algorithm
from complexity_estimator import estimate_complexity_from_features

EXPECTED = {
    3: ("BFS", "O(V + E)", "O(V)"),
    4: ("BFS", "O(V + E)", "O(V)"),
    5: ("DFS (iterative)", "O(V + E)", "O(V)"),
    6: ("DFS (iterative)", "O(V + E)", "O(V)"),
    7: ("DFS", "O(V + E)", "O(V)"),
    8: ("Dijkstra's Algorithm", "O((V + E) log V)", "O(V)"),
    9: ("Dijkstra's Algorithm", "O((V + E) log V)", "O(V)"),
    10: ("Bellman-Ford", "O(V * E)", "O(V)"),
    11: ("0-1 BFS", "O(V + E)", "O(V)"),
    12: ("All-Pairs Shortest Path (repeated Bellman-Ford)", "O(V^2 * E)", "O(V^2)"),
    13: ("Topological Sort (Kahn)", "O(V + E)", "O(V)"),
    14: ("Topological Sort (DFS)", "O(V + E)", "O(V)"),
    15: ("DAG Shortest Path", "O(V + E)", "O(V)"),
    16: ("DAG Shortest Path", "O(V + E)", "O(V)"),
}

def main():
    print("=" * 65)
    print("TESTING 14 GRAPH SAMPLES: PREDICTION & COMPLEXITY")
    print("=" * 65)
    
    passed = 0
    for num, name, code in samples:
        exp_label, exp_time, exp_space = EXPECTED[num]
        feats = extract_features(code)
        pred_label, conf = predict_algorithm(code, feats)
        
        assert pred_label == exp_label, f"Sample {num} ({name}): Expected label '{exp_label}', got '{pred_label}'"
        assert conf >= 0.70, f"Sample {num} ({name}): Expected confidence >= 0.70, got {conf}"
        
        cx = estimate_complexity_from_features(feats, algorithm=pred_label, class_confidence=conf)
        cx_time = cx.get("time_complexity") if isinstance(cx, dict) else cx.time
        cx_space = cx.get("space_complexity") if isinstance(cx, dict) else cx.space
        
        assert cx_time == exp_time, f"Sample {num} ({name}): Expected time '{exp_time}', got '{cx_time}'"
        assert cx_space == exp_space, f"Sample {num} ({name}): Expected space '{exp_space}', got '{cx_space}'"
        
        print(f"[PASS] Sample {num:2d}: {pred_label:<45} conf: {conf:.2f} | Time: {cx_time:<16} | Space: {cx_space}")
        passed += 1

    print("=" * 65)
    print(f"ALL 14 GRAPH SAMPLES PASSED: {passed}/{len(samples)}")
    print("=" * 65)

if __name__ == "__main__":
    main()

