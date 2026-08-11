T1_PERMUTATIONS = '''
def __FUNC__(__ARR__):
    # generate all permutations via backtracking with explicit undo
    __RESULT__ = []
    __PATH__ = []
    __USED__ = [False] * len(__ARR__)

    def __HELPER__():
        if len(__PATH__) == len(__ARR__):
            __RESULT__.append(__PATH__[:])
            return
        for __I__ in range(len(__ARR__)):
            if __USED__[__I__]:
                continue
            __USED__[__I__] = True
            __PATH__.append(__ARR__[__I__])
            __HELPER__()
            __PATH__.pop()
            __USED__[__I__] = False

    __HELPER__()
    return __RESULT__
'''

T2_NQUEENS = '''
def __FUNC__(__N__):
    # N-Queens via backtracking, undoing placement on dead ends
    __RESULT__ = []
    __COLS__ = set()
    __DIAG1__ = set()
    __DIAG2__ = set()
    __PLACEMENT__ = []

    def __HELPER__(__ROW__):
        if __ROW__ == __N__:
            __RESULT__.append(__PLACEMENT__[:])
            return
        for __COL__ in range(__N__):
            if __COL__ in __COLS__ or (__ROW__ - __COL__) in __DIAG1__ or (__ROW__ + __COL__) in __DIAG2__:
                continue
            __COLS__.add(__COL__)
            __DIAG1__.add(__ROW__ - __COL__)
            __DIAG2__.add(__ROW__ + __COL__)
            __PLACEMENT__.append(__COL__)
            __HELPER__(__ROW__ + 1)
            __PLACEMENT__.pop()
            __COLS__.remove(__COL__)
            __DIAG1__.remove(__ROW__ - __COL__)
            __DIAG2__.remove(__ROW__ + __COL__)

    __HELPER__(0)
    return __RESULT__
'''

T3_COMBINATION_SUM = '''
def __FUNC__(__ARR__, __TARGET__):
    # combination sum via backtracking with undo on the running path
    __RESULT__ = []
    __PATH__ = []

    def __HELPER__(__START__, __REMAINING__):
        if __REMAINING__ == 0:
            __RESULT__.append(__PATH__[:])
            return
        if __REMAINING__ < 0:
            return
        for __I__ in range(__START__, len(__ARR__)):
            __PATH__.append(__ARR__[__I__])
            __HELPER__(__I__, __REMAINING__ - __ARR__[__I__])
            __PATH__.pop()

    __HELPER__(0, __TARGET__)
    return __RESULT__
'''

T4_CONSTRAINED_PLACEMENT = '''
def __FUNC__(__ARR__, __K__):
    # subsets of size k via backtracking, explicit undo (pop) after recursion
    __RESULT__ = []
    __PATH__ = []

    def __HELPER__(__START__):
        if len(__PATH__) == __K__:
            __RESULT__.append(__PATH__[:])
            return
        for __I__ in range(__START__, len(__ARR__)):
            __PATH__.append(__ARR__[__I__])
            __HELPER__(__I__ + 1)
            __PATH__.pop()

    __HELPER__(0)
    return __RESULT__
'''

TEMPLATES = [T1_PERMUTATIONS, T2_NQUEENS, T3_COMBINATION_SUM, T4_CONSTRAINED_PLACEMENT]