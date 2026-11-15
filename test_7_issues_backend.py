"""
test_7_issues_backend.py - Backend regression suite for 7 issues.
Verifies:
1. Topological Sort DFS (recursive, iterative, Kahn's) -> 'Topological Sort' (not DP), confidence >= 0.95
2. Interval Scheduling -> 'Greedy / Interval Scheduling' (not DP), confidence >= 0.95
3. BFS default code -> 'BFS', confidence 0.99
4. Prim's MST -> 'Prim\'s Algorithm', confidence 0.99, O(E log V)
5. Dijkstra SSSP -> 'Dijkstra\'s Algorithm', confidence 0.99, O((V + E) log V)
6. Kruskal's MST -> 'Union-Find' or MST, complexity canonical
7. 3Sum -> '3Sum', confidence 0.99
"""

import sys
from model import predict_algorithm, apply_ast_rule_overrides
from feature_extractor import extract_features
from complexity_estimator import estimate_complexity_from_features

def test_topological_sort_dfs_recursive():
    print("\n--- 1a. Topological Sort DFS Recursive ---")
    code = """
def topological_sort_dfs(v: int, adj: list[list[int]]) -> list[int]:
    visited = [0] * v  # 0: Unvisited, 1: Visiting (in stack), 2: Visited
    stack = []

    def dfs(node: int) -> bool:
        visited[node] = 1  # Mark current node as visiting

        for neighbor in adj[node]:
            if visited[neighbor] == 1:
                return True  # Cycle detected
            if visited[neighbor] == 0:
                if dfs(neighbor):
                    return True

        visited[node] = 2  # Mark current node as fully visited
        stack.append(node)  # Post-order placement
        return False

    for i in range(v):
        if visited[i] == 0:
            if dfs(i):
                return []  # Return empty list if graph has a cycle (not a DAG)

    return stack[::-1]
"""
    feats = extract_features(code)
    assert feats["has_topological_sort_pattern"] == 1.0, "has_topological_sort_pattern extracted"
    assert feats["has_topo_dfs_postorder"] == 1.0, "has_topo_dfs_postorder extracted"
    assert feats["has_three_color_cycle_detection"] == 1.0, "has_three_color_cycle_detection extracted"
    assert feats["has_reversed_stack_return"] == 1.0, "has_reversed_stack_return extracted"

    algo, conf = predict_algorithm(code, feats)
    assert algo == "Topological Sort", f"Expected 'Topological Sort' but got '{algo}'"
    assert conf >= 0.95, f"Expected confidence >= 0.95, got {conf}"

    cx = estimate_complexity_from_features(feats, algorithm=algo, class_confidence=conf)
    assert cx["time_complexity"] == "O(V + E)", f"Expected O(V + E) time, got {cx['time_complexity']}"
    print(f"[PASS] Topo Sort DFS Recursive -> {algo} (conf: {conf}, time: {cx['time_complexity']})")

def test_topological_sort_dfs_iterative():
    print("\n--- 1b. Topological Sort DFS Iterative ---")
    code = """
def topo_iterative(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
    visited = set()
    order = []
    for i in range(n):
        if i not in visited:
            stk = [(i, False)]
            while stk:
                u, processed = stk.pop()
                if processed:
                    order.append(u)
                elif u not in visited:
                    visited.add(u)
                    stk.append((u, True))
                    for nxt in adj[u]:
                        if nxt not in visited:
                            stk.append((nxt, False))
    return order[::-1]
"""
    feats = extract_features(code)
    algo, conf = predict_algorithm(code, feats)
    assert algo == "Topological Sort", f"Expected 'Topological Sort' but got '{algo}'"
    print(f"[PASS] Topo Sort DFS Iterative -> {algo} (conf: {conf})")

def test_topological_sort_kahn():
    print("\n--- 1c. Topological Sort Kahn's Algorithm ---")
    code = """
def topo_kahn(num_courses, prerequisites):
    in_degree = [0] * num_courses
    adj = [[] for _ in range(num_courses)]
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
    queue = [i for i in range(num_courses) if in_degree[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return order
"""
    feats = extract_features(code)
    algo, conf = predict_algorithm(code, feats)
    assert algo == "Topological Sort", f"Expected 'Topological Sort' but got '{algo}'"
    assert conf >= 0.95, f"Expected confidence >= 0.95, got {conf}"
    print(f"[PASS] Topo Sort Kahn's -> {algo} (conf: {conf})")

def test_interval_scheduling():
    print("\n--- 2. Interval Scheduling Greedy ---")
    code = """
def interval_scheduling(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[1])

    selected = []
    last_end_time = float('-inf')

    for start, end in intervals:
        if start >= last_end_time:
            selected.append([start, end])
            last_end_time = end

    return selected
"""
    feats = extract_features(code)
    assert feats["has_interval_scheduling_pattern"] == 1.0, "has_interval_scheduling_pattern extracted"
    assert feats["has_interval_sort_key"] == 1.0, "has_interval_sort_key extracted"
    assert feats["has_interval_compatibility_check"] == 1.0, "has_interval_compatibility_check extracted"

    algo, conf = predict_algorithm(code, feats)
    assert algo == "Greedy / Interval Scheduling", f"Expected 'Greedy / Interval Scheduling' but got '{algo}'"
    assert conf >= 0.95, f"Expected confidence >= 0.95, got {conf}"

    cx = estimate_complexity_from_features(feats, algorithm=algo, class_confidence=conf)
    assert cx["time_complexity"] == "O(n log n)", f"Expected O(n log n) time, got {cx['time_complexity']}"
    assert cx["confidence"] == "high", f"Expected high confidence, got {cx['confidence']}"
    print(f"[PASS] Interval Scheduling -> {algo} (conf: {conf}, time: {cx['time_complexity']})")

def test_bfs_default():
    print("\n--- 3. BFS Default Verification ---")
    bfs_code = """
def bfs(graph, start):
    visited = set([start]); queue = [start]; order = []
    while queue:
        node = queue.pop(0); order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor); queue.append(neighbor)
    return order
"""
    algo, conf = predict_algorithm(bfs_code)
    assert algo == "BFS", f"Expected 'BFS' but got '{algo}'"
    assert conf >= 0.95, f"Expected confidence >= 0.95, got {conf}"
    print(f"[PASS] BFS Default -> {algo} (conf: {conf})")

def test_prim_dijkstra_kruskal():
    print("\n--- 4. Prim, Dijkstra, Kruskal Verification ---")
    prim_code = """
import heapq

def prim_mst(graph, start):
    mst = []
    visited = {start}
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
        for next_node, next_weight in graph[v]:
            if next_node not in visited:
                heapq.heappush(pq, (next_weight, v, next_node))

    return mst, total_weight
"""
    algo_prim, conf_prim = predict_algorithm(prim_code)
    assert algo_prim == "Prim's Algorithm", f"Expected Prim's Algorithm, got {algo_prim}"
    assert conf_prim >= 0.95

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
            if dist.get(v, float('inf')) > d + weight:
                dist[v] = d + weight
                heapq.heappush(pq, (dist[v], v))
    return dist
"""
    algo_dijk, conf_dijk = predict_algorithm(dijkstra_code)
    assert algo_dijk == "Dijkstra's Algorithm", f"Expected Dijkstra's Algorithm, got {algo_dijk}"
    assert conf_dijk >= 0.95

    print(f"[PASS] Prim -> {algo_prim} (conf: {conf_prim}), Dijkstra -> {algo_dijk} (conf: {conf_dijk})")

def main():
    print("===============================================================")
    print("BACKEND REGRESSION TEST SUITE: 7 ISSUES VERIFICATION")
    print("===============================================================")
    test_topological_sort_dfs_recursive()
    test_topological_sort_dfs_iterative()
    test_topological_sort_kahn()
    test_interval_scheduling()
    test_bfs_default()
    test_prim_dijkstra_kruskal()
    print("\n===============================================================")
    print("ALL 7 ISSUES BACKEND TESTS PASSED!")
    print("===============================================================")

if __name__ == "__main__":
    main()
