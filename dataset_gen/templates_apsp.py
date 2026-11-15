"""
Templates for All-Pairs Shortest Path (APSP) via repeated Bellman-Ford per source vertex.
"""

T1_APSP_REPEATED_BELLMAN_FORD = '''
def _bellman_ford_row(vertices, edges, src):
    dist = [float('inf')] * vertices
    dist[src] = 0
    for _ in range(vertices - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None
    return dist

def __FUNC__(vertices, edges):
    # All-Pairs Shortest Path (APSP) running Bellman-Ford for every source
    matrix = []
    for src in range(vertices):
        row = _bellman_ford_row(vertices, edges, src)
        matrix.append(row)
    return matrix
'''

T2_APSP_BELLMAN_FORD_NESTED = '''
def __FUNC__(num_nodes: int, edge_list: list[list[int]]) -> list[list[float]]:
    # Compute all-pairs shortest paths via repeated single-source Bellman-Ford
    all_pairs_matrix = []
    for source in range(num_nodes):
        dists = [float('inf')] * num_nodes
        dists[source] = 0.0
        for _ in range(num_nodes - 1):
            updated = False
            for u, v, w in edge_list:
                if dists[u] + w < dists[v]:
                    dists[v] = dists[u] + w
                    updated = True
            if not updated:
                break
        all_pairs_matrix.append(dists)
    return all_pairs_matrix
'''

T3_APSP_BELLMAN_ADJACENCY = '''
def _sssp_bellman(v_count, adj, start_v):
    d = [float('inf')] * v_count
    d[start_v] = 0
    for _ in range(v_count - 1):
        for u in range(v_count):
            for v, weight in adj.get(u, []):
                if d[u] + weight < d[v]:
                    d[v] = d[u] + weight
    return d

def __FUNC__(v_count, adj):
    # Build complete VxV shortest path distance matrix
    dist_matrix = []
    for u in range(v_count):
        row = _sssp_bellman(v_count, adj, u)
        dist_matrix.append(row)
    return dist_matrix
'''

T4_APSP_BELLMAN_CYCLE_CHECK = '''
def _bellman_pass(n, edges, s):
    dist = [float('inf')] * n
    dist[s] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return []
    return dist

def __FUNC__(n, edges):
    # APSP with repeated Bellman-Ford and negative cycle validation
    res = []
    for i in range(n):
        r = _bellman_pass(n, edges, i)
        if not r:
            return None
        res.append(r)
    return res
'''

TEMPLATES = [
    T1_APSP_REPEATED_BELLMAN_FORD,
    T2_APSP_BELLMAN_FORD_NESTED,
    T3_APSP_BELLMAN_ADJACENCY,
    T4_APSP_BELLMAN_CYCLE_CHECK,
]

