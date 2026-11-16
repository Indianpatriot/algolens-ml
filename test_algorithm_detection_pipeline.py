from complexity_estimator import estimate_complexity_from_features
from feature_extractor import extract_features
from model import predict_algorithm


TOPOLOGICAL_DFS = """
def order_dependencies(graph):
    visited = set()
    path = []
    result = []

    def visit(node):
        if node in path:
            return False
        if node in visited:
            return True
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if not visit(neighbor):
                return False
        path.pop()
        result.append(node)
        return True

    for node in graph:
        if not visit(node):
            return []
    result.reverse()
    return result
"""


DAG_LONGEST_PATH = """
def maximize_distances(graph, start, order):
    distance = {node: float("-inf") for node in graph}
    distance[start] = 0
    for current in order:
        if distance[current] != float("-inf"):
            for neighbor, weight in graph[current]:
                distance[neighbor] = max(
                    distance[neighbor], distance[current] + weight
                )
    return distance
"""


PRIM_MATRIX = """
def find_min_key(keys, visited):
    best_vertex = -1
    for candidate in range(len(keys)):
        if not visited[candidate] and (
            best_vertex == -1 or keys[candidate] < keys[best_vertex]
        ):
            best_vertex = candidate
    return best_vertex


def build_tree(matrix):
    vertex_count = len(matrix)
    keys = [float("inf")] * vertex_count
    parent = [-1] * vertex_count
    visited = [False] * vertex_count
    keys[0] = 0
    mst = []
    total_weight = 0
    for _ in range(vertex_count):
        current = find_min_key(keys, visited)
        if current == -1:
            break
        visited[current] = True
        for neighbor in range(vertex_count):
            weight = matrix[current][neighbor]
            if weight and not visited[neighbor] and weight < keys[neighbor]:
                keys[neighbor] = weight
                parent[neighbor] = current
    for node in range(1, vertex_count):
        if parent[node] != -1:
            mst.append((parent[node], node, matrix[node][parent[node]]))
            total_weight += matrix[node][parent[node]]
    return mst, total_weight
"""


PRIM_ADJACENCY_LIST = """
def find_min_key(keys, visited):
    best_vertex = -1
    for candidate in range(len(keys)):
        if not visited[candidate] and (
            best_vertex == -1 or keys[candidate] < keys[best_vertex]
        ):
            best_vertex = candidate
    return best_vertex


def build_tree(graph):
    vertex_count = len(graph)
    keys = [float("inf")] * vertex_count
    parent = [-1] * vertex_count
    visited = [False] * vertex_count
    keys[0] = 0
    mst = []
    total_weight = 0
    for _ in range(vertex_count):
        current = find_min_key(keys, visited)
        if current == -1:
            break
        visited[current] = True
        for neighbor, weight in graph[current]:
            if not visited[neighbor] and weight < keys[neighbor]:
                keys[neighbor] = weight
                parent[neighbor] = current
    for node in range(1, vertex_count):
        if parent[node] != -1:
            mst.append((parent[node], node))
            total_weight += keys[node]
    return mst, total_weight
"""


COMBINATION_SUM = """
def combination_sum(candidates, target):
    result = []
    path = []

    def backtrack(start, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] <= remaining:
                path.append(candidates[i])
                backtrack(i, remaining - candidates[i])
                path.pop()

    backtrack(0, target)
    return result
"""


def classify(code):
    features = extract_features(code)
    algorithm, confidence = predict_algorithm(code, features)
    complexity = estimate_complexity_from_features(
        features, algorithm=algorithm, class_confidence=confidence
    )
    return features, algorithm, complexity


def test_dfs_topological_sort_beats_backtracking():
    features, algorithm, complexity = classify(TOPOLOGICAL_DFS)

    assert features["has_topological_sort_dfs"] == 1.0
    assert algorithm == "Topological Sort (DFS)"
    assert complexity["time_complexity"] == "O(V + E)"


def test_combination_sum_remains_backtracking():
    features, algorithm, _ = classify(COMBINATION_SUM)

    assert features["has_topological_sort_dfs"] == 0.0
    assert "Topological Sort" not in algorithm
    assert algorithm != "Permutations"


def test_dag_longest_path_is_not_dag_shortest_path():
    features, algorithm, complexity = classify(DAG_LONGEST_PATH)

    assert features["has_dag_longest_path"] == 1.0
    assert algorithm == "DAG Longest Path"
    assert complexity["time_complexity"] == "O(V + E)"
    assert complexity["space_complexity"] == "O(V)"


def test_prim_adjacency_matrix_is_not_union_find():
    features, algorithm, complexity = classify(PRIM_MATRIX)

    assert features["has_prim_array_mst_pattern"] == 1.0
    assert features["has_prim_matrix_structure"] == 1.0
    assert algorithm == "Prim's Algorithm"
    assert complexity["time_complexity"] == "O(V^2)"
    assert complexity["space_complexity"] == "O(V^2)"


def test_prim_adjacency_list_is_not_dfs_or_union_find():
    features, algorithm, complexity = classify(PRIM_ADJACENCY_LIST)

    assert features["has_prim_array_mst_pattern"] == 1.0
    assert features["has_prim_adjacency_list_structure"] == 1.0
    assert algorithm == "Prim's Algorithm"
    assert complexity["time_complexity"] == "O(V^2 + E)"
    assert complexity["space_complexity"] == "O(V + E)"
