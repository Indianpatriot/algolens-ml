"""
Templates for Topological Sort (Kahn's in-degree queue & DFS post-order with cycle detection).
Generates diverse code samples for training the classifier.
"""

T1_TOPO_DFS_RECURSIVE_3COLOR = '''
def __FUNC__(v: int, adj: list[list[int]]) -> list[int]:
    # DFS post-order topological sort with 3-state cycle detection
    visited = [0] * v  # 0: unvisited, 1: visiting (in stack), 2: visited
    stack = []

    def dfs(node: int) -> bool:
        visited[node] = 1
        for neighbor in adj[node]:
            if visited[neighbor] == 1:
                return True
            if visited[neighbor] == 0:
                if dfs(neighbor):
                    return True
        visited[node] = 2
        stack.append(node)
        return False

    for i in range(v):
        if visited[i] == 0:
            if dfs(i):
                return []
    return stack[::-1]
'''

T2_TOPO_KAHNS_INDEGREE = '''
def __FUNC__(num_nodes: int, edges: list[list[int]]) -> list[int]:
    # Kahn's algorithm (BFS topological sort with in-degree array)
    in_degree = [0] * num_nodes
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1

    queue = [i for i in range(num_nodes) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_nodes else []
'''

T3_TOPO_DFS_NESTED_HELPER = '''
def __FUNC__(graph):
    # DFS topological sort using visited set and post-order result list
    visited = set()
    visiting = set()
    order = []

    def dfs(u):
        if u in visiting:
            return False
        if u in visited:
            return True
        visiting.add(u)
        for v in graph.get(u, []):
            if not dfs(v):
                return False
        visiting.remove(u)
        visited.add(u)
        order.append(u)
        return True

    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []
    return order[::-1]
'''

T4_TOPO_DFS_EXPLICIT_STACK = '''
def __FUNC__(n: int, prerequisites: list[list[int]]) -> list[int]:
    # Course schedule / DAG topological ordering via post-order DFS
    adj = [[] for _ in range(n)]
    for dest, src in prerequisites:
        adj[src].append(dest)

    visited = [0] * n
    res = []

    def explore(curr: int) -> bool:
        if visited[curr] == 1:
            return False
        if visited[curr] == 2:
            return True
        visited[curr] = 1
        for nxt in adj[curr]:
            if not explore(nxt):
                return False
        visited[curr] = 2
        res.append(curr)
        return True

    for i in range(n):
        if visited[i] == 0:
            if not explore(i):
                return []
    return res[::-1]
'''

T5_TOPO_DFS_CYCLE_PATH_SET = '''
def __FUNC__(graph):
    # DFS topological sort with cycle detection using path set
    visited = set()
    path = set()
    result = []
    def dfs(node):
        visited.add(node)
        path.add(node)
        for neighbor in graph.get(node, []):
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

TEMPLATES = [
    T1_TOPO_DFS_RECURSIVE_3COLOR,
    T2_TOPO_KAHNS_INDEGREE,
    T3_TOPO_DFS_NESTED_HELPER,
    T4_TOPO_DFS_EXPLICIT_STACK,
    T5_TOPO_DFS_CYCLE_PATH_SET,
]
