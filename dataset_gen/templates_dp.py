T1_MEMOIZED_1D = '''
def __FUNC__(__N__, __MEMO__=None):
    # top-down memoized DP (climbing stairs style)
    if __MEMO__ is None:
        __MEMO__ = {}
    if __N__ <= 1:
        return 1
    if __N__ in __MEMO__:
        return __MEMO__[__N__]
    __RESULT__ = __FUNC__(__N__ - 1, __MEMO__) + __FUNC__(__N__ - 2, __MEMO__)
    __MEMO__[__N__] = __RESULT__
    return __RESULT__
'''

T2_TABULATED_1D = '''
def __FUNC__(__TARGET__, __ARR__):
    # bottom-up tabulated DP (coin change style)
    __DP__ = [float('inf')] * (__TARGET__ + 1)
    __DP__[0] = 0
    for __I__ in range(1, __TARGET__ + 1):
        for __COIN__ in __ARR__:
            if __COIN__ <= __I__:
                __DP__[__I__] = min(__DP__[__I__], __DP__[__I__ - __COIN__] + 1)
    return __DP__[__TARGET__] if __DP__[__TARGET__] != float('inf') else -1
'''

T3_2D_LCS = '''
def __FUNC__(__ARR__, __ARR2__):
    # 2D tabulated DP - longest common subsequence
    __N__ = len(__ARR__)
    __M__ = len(__ARR2__)
    __DP__ = [[0] * (__M__ + 1) for _ in range(__N__ + 1)]
    for __I__ in range(1, __N__ + 1):
        for __J__ in range(1, __M__ + 1):
            if __ARR__[__I__ - 1] == __ARR2__[__J__ - 1]:
                __DP__[__I__][__J__] = __DP__[__I__ - 1][__J__ - 1] + 1
            else:
                __DP__[__I__][__J__] = max(__DP__[__I__ - 1][__J__], __DP__[__I__][__J__ - 1])
    return __DP__[__N__][__M__]
'''

T4_2D_KNAPSACK = '''
def __FUNC__(__ARR__, __ARR2__, __TARGET__):
    # 0/1 knapsack, 2D tabulated DP
    __N__ = len(__ARR__)
    __DP__ = [[0] * (__TARGET__ + 1) for _ in range(__N__ + 1)]
    for __I__ in range(1, __N__ + 1):
        for __J__ in range(__TARGET__ + 1):
            __DP__[__I__][__J__] = __DP__[__I__ - 1][__J__]
            if __ARR__[__I__ - 1] <= __J__:
                __CANDIDATE__ = __DP__[__I__ - 1][__J__ - __ARR__[__I__ - 1]] + __ARR2__[__I__ - 1]
                __DP__[__I__][__J__] = max(__DP__[__I__][__J__], __CANDIDATE__)
    return __DP__[__N__][__TARGET__]
'''

TEMPLATES = [T1_MEMOIZED_1D, T2_TABULATED_1D, T3_2D_LCS, T4_2D_KNAPSACK]