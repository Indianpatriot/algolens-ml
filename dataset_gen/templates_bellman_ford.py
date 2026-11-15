"""
Templates for Bellman-Ford single-source shortest path with negative edge support
and negative cycle detection passes.
"""

T1_BELLMAN_FORD_STANDARD = '''
def __FUNC__(vertices, edges, start):
    # Bellman-Ford algorithm relaxing all edges V-1 times
    dist = [float('inf')] * vertices
    dist[start] = 0
    for _ in range(vertices - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    # Negative cycle detection pass
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None
    return dist
'''

T2_BELLMAN_FORD_ADJACENCY = '''
def __FUNC__(num_nodes, graph, source):
    # Bellman-Ford on adjacency list with relaxation and negative cycle check
    distances = {i: float('inf') for i in range(num_nodes)}
    distances[source] = 0
    for iteration in range(num_nodes - 1):
        changed = False
        for u in range(num_nodes):
            if distances[u] == float('inf'):
                continue
            for v, weight in graph.get(u, []):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    changed = True
        if not changed:
            break
    for u in range(num_nodes):
        for v, weight in graph.get(u, []):
            if distances[u] + weight < distances[v]:
                raise ValueError("Negative cycle detected")
    return distances
'''

T3_BELLMAN_FORD_EDGE_LIST = '''
def __FUNC__(v: int, edge_list: list[list[int]], src: int) -> list[float]:
    # Single-source shortest path with negative weight edges
    d = [float('inf')] * v
    d[src] = 0.0
    for i in range(v - 1):
        has_change = False
        for u, dest, cost in edge_list:
            if d[u] + cost < d[dest]:
                d[dest] = d[u] + cost
                has_change = True
        if not has_change:
            break
    for u, dest, cost in edge_list:
        if d[u] + cost < d[dest]:
            return []
    return d
'''

T4_BELLMAN_FORD_PREDECESSOR = '''
def __FUNC__(vertices, edges, source):
    # Bellman-Ford tracking predecessor parent pointers
    dist = [float('inf')] * vertices
    parent = [-1] * vertices
    dist[source] = 0
    for step in range(vertices - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None, None
    return dist, parent
'''

TEMPLATES = [
    T1_BELLMAN_FORD_STANDARD,
    T2_BELLMAN_FORD_ADJACENCY,
    T3_BELLMAN_FORD_EDGE_LIST,
    T4_BELLMAN_FORD_PREDECESSOR,
]

