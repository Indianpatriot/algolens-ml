import sys
from feature_extractor import extract_features
from model import predict_algorithm

sample_3 = '''
from collections import deque
graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
    return order
'''

sample_4 = '''
def bfs_bounded(start):
    visited = set([start])
    queue = [start]
    order = []
    for _ in range(len(graph)):
        if not queue:
            break
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
'''

sample_5 = '''
def dfs_iterative_reversed(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
'''

sample_6 = '''
def dfs_iterative_extend(graph, start):
    visited = set()
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            stack.extend(graph[node])
    return order
'''

sample_7 = '''
def dfs_recursive(graph):
    visited = []
    def explore(node):
        visited.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                explore(neighbor)
    for v in graph:
        if v not in visited:
            explore(v)
    return visited
'''

sample_8 = '''
import heapq
def dijkstra_lazy(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            if dist[node] + weight < dist[neighbor]:
                dist[neighbor] = dist[node] + weight
                heapq.heappush(pq, (dist[neighbor], neighbor))
    return dist
'''

sample_9 = '''
import heapq
def dijkstra_visited(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    visited = set()
    pq = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited and dist[node] + weight < dist[neighbor]:
                dist[neighbor] = dist[node] + weight
                heapq.heappush(pq, (dist[neighbor], neighbor))
    return dist
'''

sample_10 = '''
def bellman_ford(vertices, edges, start):
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
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None # negative cycle detected
    return dist
'''

sample_11 = '''
from collections import deque
def zero_one_bfs(graph, start, vertices):
    dist = [float('inf')] * vertices
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                if weight == 0:
                    queue.appendleft(v)
                else:
                    queue.append(v)
    return dist
'''

sample_12 = '''
def bellman_ford_helper(vertices, edges, src):
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

def apsp(vertices, edges):
    matrix = []
    for src in range(vertices):
        row = bellman_ford_helper(vertices, edges, src)
        matrix.append(row)
    return matrix
'''

sample_13 = '''
from collections import deque
def topological_sort_kahn(num_nodes, edges):
    in_degree = [0] * num_nodes
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(order) != num_nodes:
        return [] # cycle detected
    return order
'''

sample_14 = '''
def topological_sort_dfs_cycle(graph):
    visited = set()
    path = set()
    result = []
    def dfs(node):
        visited.add(node)
        path.add(node)
        for neighbor in graph[node]:
            if neighbor in path:
                return False
            if neighbor not in visited:
                if not dfs(neighbor):
                    return False
        path.remove(node)
        result.append(node)
        return True
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []
    return result[::-1]
'''

sample_15 = '''
def dag_shortest_path_dfs(graph, start, vertices):
    visited = set()
    topo_order = []
    def dfs(u):
        visited.add(u)
        for v, _ in graph[u]:
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
            for v, weight in graph[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
    return dist
'''

sample_16 = '''
from collections import deque
def dag_shortest_path_kahn(vertices, edges, start):
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
        return None
    dist = [float('inf')] * vertices
    dist[start] = 0
    for u in topo:
        if dist[u] != float('inf'):
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    return dist
'''

samples = [
    (3, "BFS (deque, visited on pop)", sample_3),
    (4, "BFS (bounded for-loop)", sample_4),
    (5, "Iterative DFS (reversed)", sample_5),
    (6, "Iterative DFS (extend)", sample_6),
    (7, "Recursive DFS (nested explore)", sample_7),
    (8, "Dijkstra (lazy)", sample_8),
    (9, "Dijkstra (visited set)", sample_9),
    (10, "Bellman-Ford", sample_10),
    (11, "0-1 BFS", sample_11),
    (12, "APSP (repeated Bellman-Ford)", sample_12),
    (13, "Topological Sort (Kahn)", sample_13),
    (14, "Topological Sort (DFS cycle)", sample_14),
    (15, "DAG Shortest Path (DFS)", sample_15),
    (16, "DAG Shortest Path (Kahn)", sample_16),
]

for num, name, code in samples:
    pred, conf = predict_algorithm(code)
    print(f"Sample {num}: {name} -> Predicted: {pred} ({conf})")

