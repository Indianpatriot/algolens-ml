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

T5_UNWEIGHTED_SHORTEST_PATH = '''
from collections import deque

def __FUNC__(__GRAPH__, __START__):
    # Unweighted shortest path using BFS queue and distance map
    __DIST__ = {__START__: 0}
    __QUEUE__ = deque([__START__])

    while __QUEUE__:
        __U__ = __QUEUE__.popleft()
        for __V__ in __GRAPH__.get(__U__, []):
            if __V__ not in __DIST__:
                __DIST__[__V__] = __DIST__[__U__] + 1
                __QUEUE__.append(__V__)

    return __DIST__
'''

T6_LIST_QUEUE_TRAVERSAL = '''
def __FUNC__(__GRAPH__, __START__):
    # BFS using standard list-based FIFO queue with pop(0)
    __VISITED__ = set([__START__])
    __QUEUE__ = [__START__]
    __RESULT__ = []
    while __QUEUE__:
        __NODE__ = __QUEUE__.pop(0)
        __RESULT__.append(__NODE__)
        for __NEIGHBOR__ in __GRAPH__.get(__NODE__, []):
            if __NEIGHBOR__ not in __VISITED__:
                __VISITED__.add(__NEIGHBOR__)
                __QUEUE__.append(__NEIGHBOR__)
    return __RESULT__
'''

T7_LIST_QUEUE_DIRECT_INDEX = '''
def __FUNC__(__GRAPH__, __START__):
    # BFS using list queue with direct subscript adjacency iteration
    __VISITED__ = set([__START__])
    __QUEUE__ = [__START__]
    __RESULT__ = []
    while __QUEUE__:
        __NODE__ = __QUEUE__.pop(0)
        __RESULT__.append(__NODE__)
        for __NEIGHBOR__ in __GRAPH__[__NODE__]:
            if __NEIGHBOR__ not in __VISITED__:
                __VISITED__.add(__NEIGHBOR__)
                __QUEUE__.append(__NEIGHBOR__)
    return __RESULT__
'''

T8_LIST_QUEUE_SHORTEST_PATH = '''
def __FUNC__(__GRAPH__, __START__, __TARGET__):
    # BFS shortest path in unweighted graph using list queue
    __VISITED__ = {__START__}
    __QUEUE__ = [(__START__, 0)]
    while __QUEUE__:
        __NODE__, __DIFF__ = __QUEUE__.pop(0)
        if __NODE__ == __TARGET__:
            return __DIFF__
        for __NEIGHBOR__ in __GRAPH__.get(__NODE__, []):
            if __NEIGHBOR__ not in __VISITED__:
                __VISITED__.add(__NEIGHBOR__)
                __QUEUE__.append((__NEIGHBOR__, __DIFF__ + 1))
    return -1
'''

T9_GRID_BFS_LIST = '''
def __FUNC__(__ARR__, __START__):
    # Grid BFS using list queue
    __ROWS__ = len(__ARR__)
    __COLS__ = len(__ARR__[0]) if __ROWS__ else 0
    __VISITED__ = set([__START__])
    __QUEUE__ = [__START__]
    while __QUEUE__:
        __R__, __C__ = __QUEUE__.pop(0)
        for __DR__, __DC__ in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            __NR__, __NC__ = __R__ + __DR__, __C__ + __DC__
            if 0 <= __NR__ < __ROWS__ and 0 <= __NC__ < __COLS__:
                if (__NR__, __NC__) not in __VISITED__:
                    __VISITED__.add((__NR__, __NC__))
                    __QUEUE__.append((__NR__, __NC__))
    return __VISITED__
'''

T10_TREE_LEVEL_ORDER_LIST = '''
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def __FUNC__(root):
    # Tree level order BFS using list queue
    if not root:
        return []
    __QUEUE__ = [root]
    __RESULT__ = []
    while __QUEUE__:
        __NODE__ = __QUEUE__.pop(0)
        __RESULT__.append(__NODE__.val)
        if __NODE__.left:
            __QUEUE__.append(__NODE__.left)
        if __NODE__.right:
            __QUEUE__.append(__NODE__.right)
    return __RESULT__
'''

TEMPLATES = [
    T1_GRAPH_TRAVERSAL,
    T2_SHORTEST_PATH,
    T3_TREE_LEVEL_ORDER,
    T4_GRID_BFS,
    T5_UNWEIGHTED_SHORTEST_PATH,
    T6_LIST_QUEUE_TRAVERSAL,
    T7_LIST_QUEUE_DIRECT_INDEX,
    T8_LIST_QUEUE_SHORTEST_PATH,
    T9_GRID_BFS_LIST,
    T10_TREE_LEVEL_ORDER_LIST,
]