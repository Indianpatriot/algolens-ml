T1_GRAPH_RECURSIVE = '''
def __FUNC__(__GRAPH__, __NODE__, __VISITED__=None):
    # recursive DFS over an adjacency-list graph
    if __VISITED__ is None:
        __VISITED__ = set()
    __VISITED__.add(__NODE__)
    __RESULT__ = [__NODE__]
    for __NEIGHBOR__ in __GRAPH__.get(__NODE__, []):
        if __NEIGHBOR__ not in __VISITED__:
            __RESULT__.extend(__FUNC__(__GRAPH__, __NEIGHBOR__, __VISITED__))
    return __RESULT__
'''

T2_GRAPH_ITERATIVE_STACK = '''
def __FUNC__(__GRAPH__, __START__):
    # iterative DFS using an explicit stack
    __VISITED__ = set()
    __STACK__ = [__START__]
    __RESULT__ = []
    while __STACK__:
        __NODE__ = __STACK__.pop()
        if __NODE__ in __VISITED__:
            continue
        __VISITED__.add(__NODE__)
        __RESULT__.append(__NODE__)
        for __NEIGHBOR__ in __GRAPH__.get(__NODE__, []):
            if __NEIGHBOR__ not in __VISITED__:
                __STACK__.append(__NEIGHBOR__)
    return __RESULT__
'''

T3_TREE_PREORDER = '''
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def __FUNC__(root):
    # recursive preorder DFS traversal of a binary tree
    __RESULT__ = []
    def __HELPER__(__NODE__):
        if __NODE__ is None:
            return
        __RESULT__.append(__NODE__.val)
        __HELPER__(__NODE__.left)
        __HELPER__(__NODE__.right)
    __HELPER__(root)
    return __RESULT__
'''

T4_GRID_FLOODFILL = '''
def __FUNC__(__ARR__, __START__):
    # DFS flood fill / connected components on a 2D grid
    __ROWS__ = len(__ARR__)
    __COLS__ = len(__ARR__[0]) if __ROWS__ else 0
    __VISITED__ = [[False] * __COLS__ for _ in range(__ROWS__)]

    def __HELPER__(__R__, __C__):
        if __R__ < 0 or __R__ >= __ROWS__ or __C__ < 0 or __C__ >= __COLS__:
            return
        if __VISITED__[__R__][__C__] or __ARR__[__R__][__C__] == 0:
            return
        __VISITED__[__R__][__C__] = True
        __HELPER__(__R__ + 1, __C__)
        __HELPER__(__R__ - 1, __C__)
        __HELPER__(__R__, __C__ + 1)
        __HELPER__(__R__, __C__ - 1)

    __I__, __J__ = __START__
    __HELPER__(__I__, __J__)
    return __VISITED__
'''

TEMPLATES = [T1_GRAPH_RECURSIVE, T2_GRAPH_ITERATIVE_STACK, T3_TREE_PREORDER, T4_GRID_FLOODFILL]