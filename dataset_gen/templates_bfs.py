from collections import deque

T1_GRAPH_TRAVERSAL = '''
from collections import deque

def __FUNC__(__GRAPH__, __START__):
    # BFS over an adjacency-list graph, returns visited order
    __VISITED__ = set([__START__])
    __QUEUE__ = deque([__START__])
    __RESULT__ = []
    while __QUEUE__:
        __NODE__ = __QUEUE__.popleft()
        __RESULT__.append(__NODE__)
        for __NEIGHBOR__ in __GRAPH__.get(__NODE__, []):
            if __NEIGHBOR__ not in __VISITED__:
                __VISITED__.add(__NEIGHBOR__)
                __QUEUE__.append(__NEIGHBOR__)
    return __RESULT__
'''

T2_SHORTEST_PATH = '''
from collections import deque

def __FUNC__(__GRAPH__, __START__, __TARGET__):
    # BFS shortest path length in an unweighted graph
    __VISITED__ = {__START__}
    __QUEUE__ = deque([(__START__, 0)])
    while __QUEUE__:
        __NODE__, __DIFF__ = __QUEUE__.popleft()
        if __NODE__ == __TARGET__:
            return __DIFF__
        for __NEIGHBOR__ in __GRAPH__.get(__NODE__, []):
            if __NEIGHBOR__ not in __VISITED__:
                __VISITED__.add(__NEIGHBOR__)
                __QUEUE__.append((__NEIGHBOR__, __DIFF__ + 1))
    return -1
'''

T3_TREE_LEVEL_ORDER = '''
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def __FUNC__(root):
    # level-order traversal of a binary tree using BFS
    if root is None:
        return []
    __RESULT__ = []
    __QUEUE__ = deque([root])
    while __QUEUE__:
        __N__ = len(__QUEUE__)
        __LEVEL__ = []
        for __I__ in range(__N__):
            __NODE__ = __QUEUE__.popleft()
            __LEVEL__.append(__NODE__.val)
            if __NODE__.left:
                __QUEUE__.append(__NODE__.left)
            if __NODE__.right:
                __QUEUE__.append(__NODE__.right)
        __RESULT__.append(__LEVEL__)
    return __RESULT__
'''

T4_GRID_BFS = '''
from collections import deque

def __FUNC__(__ARR__, __START__):
    # multi-source-capable BFS over a 2D grid, returns distance grid
    __ROWS__ = len(__ARR__)
    __COLS__ = len(__ARR__[0]) if __ROWS__ else 0
    __VISITED__ = [[False] * __COLS__ for _ in range(__ROWS__)]
    __QUEUE__ = deque([__START__])
    __I__, __J__ = __START__
    __VISITED__[__I__][__J__] = True
    __DIST__ = [[-1] * __COLS__ for _ in range(__ROWS__)]
    __DIST__[__I__][__J__] = 0
    while __QUEUE__:
        __R__, __C__ = __QUEUE__.popleft()
        for __DR__, __DC__ in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            __NR__, __NC__ = __R__ + __DR__, __C__ + __DC__
            if 0 <= __NR__ < __ROWS__ and 0 <= __NC__ < __COLS__ and not __VISITED__[__NR__][__NC__]:
                __VISITED__[__NR__][__NC__] = True
                __DIST__[__NR__][__NC__] = __DIST__[__R__][__C__] + 1
                __QUEUE__.append((__NR__, __NC__))
    return __DIST__
'''

T5_DIJKSTRA_PQ = '''
import heapq

def __FUNC__(__GRAPH__, __START__):
    # Dijkstra single-source shortest path using priority queue
    __DIST__ = {__NODE__: float('inf') for __NODE__ in __GRAPH__}
    __DIST__[__START__] = 0
    __PQ__ = [(0, __START__)]

    while __PQ__:
        __CURR_D__, __U__ = heapq.heappop(__PQ__)
        if __CURR_D__ > __DIST__[__U__]:
            continue
        for __V__, __WEIGHT__ in __GRAPH__.get(__U__, []):
            if __DIST__[__U__] + __WEIGHT__ < __DIST__[__V__]:
                __DIST__[__V__] = __DIST__[__U__] + __WEIGHT__
                heapq.heappush(__PQ__, (__DIST__[__V__], __V__))

    return __DIST__
'''

TEMPLATES = [T1_GRAPH_TRAVERSAL, T2_SHORTEST_PATH, T3_TREE_LEVEL_ORDER, T4_GRID_BFS, T5_DIJKSTRA_PQ]