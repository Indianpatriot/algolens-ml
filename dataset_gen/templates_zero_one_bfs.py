"""
Templates for 0-1 BFS using deque (appendleft on weight 0, append on weight 1).
"""

T1_ZERO_ONE_BFS_STANDARD = '''
from collections import deque

def __FUNC__(graph, start, vertices):
    # 0-1 BFS on weighted graph with weights in {0, 1}
    dist = [float('inf')] * vertices
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, weight in graph.get(u, []):
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                if weight == 0:
                    queue.appendleft(v)
                else:
                    queue.append(v)
    return dist
'''

T2_ZERO_ONE_BFS_GRID = '''
from collections import deque

def __FUNC__(grid, start_r, start_c):
    # 0-1 BFS on grid with 0/1 transition costs
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[start_r][start_c] = 0
    dq = deque([(start_r, start_c)])
    while dq:
        r, c = dq.popleft()
        for dr, dc, weight in [(0, 1, 0), (1, 0, 1), (0, -1, 0), (-1, 0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if dist[r][c] + weight < dist[nr][nc]:
                    dist[nr][nc] = dist[r][c] + weight
                    if weight == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))
    return dist
'''

T3_ZERO_ONE_BFS_TUPLES = '''
from collections import deque

def __FUNC__(n, adj, src):
    # 0-1 BFS single source shortest path using deque front/back pushes
    distance = {i: float('inf') for i in range(n)}
    distance[src] = 0
    frontier = deque([src])
    while frontier:
        curr = frontier.popleft()
        for neighbor, cost in adj[curr]:
            new_cost = distance[curr] + cost
            if new_cost < distance[neighbor]:
                distance[neighbor] = new_cost
                if cost == 0:
                    frontier.appendleft(neighbor)
                else:
                    frontier.append(neighbor)
    return distance
'''

T4_ZERO_ONE_BFS_DIRECTED = '''
from collections import deque

def __FUNC__(v_count: int, edges: list[tuple[int, int, int]], source_node: int) -> list[int]:
    # 0-1 BFS shortest paths on directed graph
    adj_list = [[] for _ in range(v_count)]
    for u, v, w in edges:
        adj_list[u].append((v, w))
    distances = [10**9] * v_count
    distances[source_node] = 0
    q = deque([source_node])
    while q:
        u = q.popleft()
        for v, w in adj_list[u]:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                if w == 0:
                    q.appendleft(v)
                else:
                    q.append(v)
    return distances
'''

TEMPLATES = [
    T1_ZERO_ONE_BFS_STANDARD,
    T2_ZERO_ONE_BFS_GRID,
    T3_ZERO_ONE_BFS_TUPLES,
    T4_ZERO_ONE_BFS_DIRECTED,
]
