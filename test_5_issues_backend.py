"""
test_5_issues_backend.py
Verification suite for Issue 4 (Prim's MST vs BFS/Dijkstra) and backend stability for the 5 issues.
"""

from model import predict_algorithm
from complexity_estimator import estimate_complexity
from feature_extractor import extract_features

def test_prim_mst():
    print("\n--- 1. Prim MST Backend ---")
    prim_code = """
import heapq

def prim_mst(graph, start=0):
    visited = set([start])
    mst = []
    total_weight = 0
    pq = []
    for neighbor, weight in graph[start]:
        heapq.heappush(pq, (weight, start, neighbor))
    
    while pq:
        weight, u, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, weight))
        total_weight += weight
        for neighbor, w in graph[v]:
            if neighbor not in visited:
                heapq.heappush(pq, (w, v, neighbor))
    return mst, total_weight
"""
    features = extract_features(prim_code)
    assert features["has_priority_queue_traversal"] == 1.0, "has_priority_queue_traversal extracted"
    assert features["has_weighted_adjacency"] == 1.0, "has_weighted_adjacency extracted"
    assert features["has_mst_weight_accumulation"] == 1.0, "has_mst_weight_accumulation extracted"
    assert features["has_mst_edge_building"] == 1.0, "has_mst_edge_building extracted"
    assert features["has_lazy_deletion_on_pop"] == 1.0, "has_lazy_deletion_on_pop extracted"
    assert features["has_dist_relaxation"] == 0.0, "has_dist_relaxation is 0 for Prim"
    print("[PASS] Prim discriminative features extracted accurately")

    algo, conf = predict_algorithm(prim_code, features=features)
    assert algo == "Prim's Algorithm", f"Prim predicted as 'Prim's Algorithm' (got '{algo}')"
    assert conf >= 0.95, f"Prim confidence >= 0.95 (got {conf})"
    print(f"[PASS] Prim algorithm classified as '{algo}' with confidence {conf}")

    comp = estimate_complexity(prim_code, algorithm=algo, class_confidence=conf)
    assert comp["time_complexity"] == "O(E log V)", f"Time complexity is O(E log V) (got '{comp['time_complexity']}')"
    assert comp["space_complexity"] == "O(V + E)", f"Space complexity is O(V + E) (got '{comp['space_complexity']}')"
    assert comp["confidence"] == "high", f"Complexity confidence is 'high' (got '{comp['confidence']}')"
    print(f"[PASS] Prim canonical complexity derived: {comp['time_complexity']}, {comp['space_complexity']}, {comp['confidence']}")

def test_prim_with_q_name():
    print("\n--- 2. Prim with 'q' queue variable (BFS collision test) ---")
    prim_q_code = """
import heapq

def find_tree(graph, s=0):
    visited = set([s])
    res = []
    total = 0
    q = []
    for to, c in graph[s]:
        heapq.heappush(q, (c, s, to))
    while q:
        cost, u, v = heapq.heappop(q)
        if v in visited:
            continue
        visited.add(v)
        res.append((u, v))
        total += cost
        for to, c in graph[v]:
            if to not in visited:
                heapq.heappush(q, (c, v, to))
    return res, total
"""
    algo, conf = predict_algorithm(prim_q_code)
    assert algo == "Prim's Algorithm", f"Prim with 'q' variable is classified as 'Prim's Algorithm' NOT 'BFS' (got '{algo}')"
    assert conf >= 0.95, f"Confidence >= 0.95 (got {conf})"
    print(f"[PASS] Prim with 'q' queue variable resolved to '{algo}' (conf={conf})")

def test_dijkstra_separation():
    print("\n--- 3. Dijkstra SSSP Separation from Prim ---")
    dijkstra_code = """
import heapq

def dijkstra(graph, start):
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if d + weight < dist.get(v, float('inf')):
                dist[v] = d + weight
                heapq.heappush(pq, (dist[v], v))
    return dist
"""
    features = extract_features(dijkstra_code)
    assert features["has_dist_relaxation"] == 1.0, "has_dist_relaxation extracted for Dijkstra"
    assert features["has_mst_weight_accumulation"] == 0.0, "has_mst_weight_accumulation is 0 for Dijkstra"

    algo, conf = predict_algorithm(dijkstra_code, features=features)
    assert algo == "Dijkstra's Algorithm", f"Dijkstra classified as 'Dijkstra's Algorithm' (got '{algo}')"
    comp = estimate_complexity(dijkstra_code, algorithm=algo, class_confidence=conf)
    assert comp["time_complexity"] == "O((V + E) log V)", f"Dijkstra time complexity is O((V + E) log V) (got '{comp['time_complexity']}')"
    assert comp["confidence"] == "high", f"Dijkstra complexity confidence is high (got '{comp['confidence']}')"
    print(f"[PASS] Dijkstra correctly separated from Prim: {algo} ({comp['time_complexity']})")

def test_3sum_backend():
    print("\n--- 4. 3Sum Backend ---")
    three_sum_code = """
def three_sum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            if total == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]: l += 1
                while l < r and nums[r] == nums[r - 1]: r -= 1
                l += 1; r -= 1
            elif total < 0:
                l += 1
            else:
                r -= 1
    return res
"""
    algo, conf = predict_algorithm(three_sum_code)
    assert algo == "3Sum", f"3Sum classified as '3Sum' (got '{algo}')"
    print(f"[PASS] 3Sum classified as '{algo}' (conf={conf})")

if __name__ == "__main__":
    test_prim_mst()
    test_prim_with_q_name()
    test_dijkstra_separation()
    test_3sum_backend()
    print("\n===============================================================")
    print("ALL BACKEND 5 ISSUES TESTS PASSED!")
    print("===============================================================")
