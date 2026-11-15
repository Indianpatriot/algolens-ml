"""
Templates for DAG Shortest Path (topological ordering followed by linear relaxation).
"""

T1_DAG_SP_DFS_TOPO = '''
def __FUNC__(graph, start, vertices):
    # DAG Single-Source Shortest Path using DFS topological sort
    visited = set()
    topo_order = []
    def dfs(u):
        visited.add(u)
        for v, _ in graph.get(u, []):
            if v not in visited:
                dfs(v)
        topo_order.append(u)
    for node in range(vertices):
        if node not in visited:
            dfs(node)
    topo_order.reverse()
    dist = [float('inf')] * vertices
    dist[start] = 0
    for u in topo_order:
        if dist[u] != float('inf'):
            for v, weight in graph.get(u, []):
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
    return dist
'''

T2_DAG_SP_KAHN_TOPO = '''
from collections import deque

def __FUNC__(vertices, edges, start):
    # DAG Shortest Path using Kahn's topological sort and edge relaxation
    in_degree = [0] * vertices
    adj = [[] for _ in range(vertices)]
    for u, v, w in edges:
        adj[u].append((v, w))
        in_degree[v] += 1
    queue = deque([i for i in range(vertices) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v, _ in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(topo) < vertices:
        return None  # Cycle detected
    dist = [float('inf')] * vertices
    dist[start] = 0
    for u in topo:
        if dist[u] != float('inf'):
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    return dist
'''

T3_DAG_SP_WEIGHTED_ADJ = '''
def __FUNC__(n: int, adj_list: list[list[tuple[int, int]]], src: int) -> list[int]:
    # Single-source shortest path in a Directed Acyclic Graph
    visited = [False] * n
    linear_order = []
    def explore(curr):
        visited[curr] = True
        for nxt, _ in adj_list[curr]:
            if not visited[nxt]:
                explore(nxt)
        linear_order.append(curr)
    for i in range(n):
        if not visited[i]:
            explore(i)
    linear_order.reverse()
    shortest = [10**9] * n
    shortest[src] = 0
    for u in linear_order:
        if shortest[u] < 10**9:
            for v, cost in adj_list[u]:
                if shortest[u] + cost < shortest[v]:
                    shortest[v] = shortest[u] + cost
    return shortest
'''

T4_DAG_LONGEST_PATH = '''
def __FUNC__(v_num, edges_weighted, source_vertex):
    # DAG Shortest Path relaxing along topological stack order
    in_deg = [0] * v_num
    graph = {i: [] for i in range(v_num)}
    for u, v, wt in edges_weighted:
        graph[u].append((v, wt))
        in_deg[v] += 1
    q = [i for i in range(v_num) if in_deg[i] == 0]
    topological_sequence = []
    while q:
        curr = q.pop(0)
        topological_sequence.append(curr)
        for nbr, _ in graph[curr]:
            in_deg[nbr] -= 1
            if in_deg[nbr] == 0:
                q.append(nbr)
    dists = [float('inf')] * v_num
    dists[source_vertex] = 0
    for u in topological_sequence:
        if dists[u] != float('inf'):
            for v, w in graph[u]:
                if dists[u] + w < dists[v]:
                    dists[v] = dists[u] + w
    return dists
'''

TEMPLATES = [
    T1_DAG_SP_DFS_TOPO,
    T2_DAG_SP_KAHN_TOPO,
    T3_DAG_SP_WEIGHTED_ADJ,
    T4_DAG_LONGEST_PATH,
]

