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

T5_LIS_1D = '''
def __FUNC__(__ARR__):
    # Longest Increasing Subsequence 1D DP table fill
    if not __ARR__:
        return 0
    __N__ = len(__ARR__)
    __DP__ = [1] * __N__
    for __I__ in range(1, __N__):
        for __J__ in range(__I__):
            if __ARR__[__J__] < __ARR__[__I__]:
                __DP__[__I__] = max(__DP__[__I__], __DP__[__J__] + 1)
    return max(__DP__)
'''

T6_HOUSE_ROBBER_1D = '''
def __FUNC__(__ARR__):
    # House Robber 1D DP table fill
    if not __ARR__:
        return 0
    __N__ = len(__ARR__)
    if __N__ == 1:
        return __ARR__[0]
    __DP__ = [0] * __N__
    __DP__[0] = __ARR__[0]
    __DP__[1] = max(__ARR__[0], __ARR__[1])
    for __I__ in range(2, __N__):
        __DP__[__I__] = max(__DP__[__I__ - 1], __DP__[__I__ - 2] + __ARR__[__I__])
    return __DP__[-1]
'''

T7_WORD_BREAK_CLASSIC = '''
def __FUNC__(s, words):
    # Word break classic 1D boolean dynamic programming
    word_set = set(words)
    __N__ = len(s)
    __DP__ = [False] * (__N__ + 1)
    __DP__[0] = True
    for __I__ in range(1, __N__ + 1):
        for __J__ in range(__I__):
            if __DP__[__J__] and s[__J__:__I__] in word_set:
                __DP__[__I__] = True
                break
    return __DP__[__N__]
'''

T8_WORD_BREAK_WORD_BOUNDED = '''
def __FUNC__(text, word_dict):
    # Word break with inner loop bounded by word lengths
    word_set = set(word_dict)
    lens = set(len(w) for w in word_set)
    __N__ = len(text)
    __DP__ = [False] * (__N__ + 1)
    __DP__[0] = True
    for __I__ in range(1, __N__ + 1):
        for l in lens:
            if __I__ >= l and __DP__[__I__ - l] and text[__I__ - l:__I__] in word_set:
                __DP__[__I__] = True
                break
    return __DP__[__N__]
'''

T9_WORD_BREAK_MEMO_RECURSION = '''
def __FUNC__(s, word_dict):
    # Word break via top-down memoized recursion
    words = set(word_dict)
    __MEMO__ = {}
    def can_segment(start_idx):
        if start_idx == len(s):
            return True
        if start_idx in __MEMO__:
            return __MEMO__[start_idx]
        for end_idx in range(start_idx + 1, len(s) + 1):
            if s[start_idx:end_idx] in words and can_segment(end_idx):
                __MEMO__[start_idx] = True
                return True
        __MEMO__[start_idx] = False
        return False
    return can_segment(0)
'''

T10_WORD_BREAK_ALT_NAMES = '''
def __FUNC__(string_val, vocab):
    # Word break using alternate variable names
    vocab_set = set(vocab)
    length = len(string_val)
    valid = [False] * (length + 1)
    valid[0] = True
    for idx in range(1, length + 1):
        for jdx in range(idx):
            if valid[jdx] and string_val[jdx:idx] in vocab_set:
                valid[idx] = True
                break
    return valid[length]
'''

T11_WORD_BREAK_FORWARD_LOOP = '''
def __FUNC__(s, word_dict):
    # Word break forward reachability loop
    word_set = set(word_dict)
    n = len(s)
    __DP__ = [True] + [False] * n
    for __I__ in range(n):
        if not __DP__[__I__]:
            continue
        for __J__ in range(__I__ + 1, n + 1):
            if s[__I__:__J__] in word_set:
                __DP__[__J__] = True
    return __DP__[n]
'''

T12_WORD_BREAK_MAX_LEN_BOUND = '''
def __FUNC__(s, word_dict):
    # Word break bounded by max word length
    words = set(word_dict)
    max_len = max(len(w) for w in words) if words else 0
    __N__ = len(s)
    __DP__ = [False] * (__N__ + 1)
    __DP__[0] = True
    for __I__ in range(1, __N__ + 1):
        for __J__ in range(max(0, __I__ - max_len), __I__):
            if __DP__[__J__] and s[__J__:__I__] in words:
                __DP__[__I__] = True
                break
    return __DP__[__N__]
'''

T13_WORD_BREAK_SUBPROBLEM_LIST = '''
def __FUNC__(s, word_dict):
    # Word break prefix table subproblems
    dict_set = set(word_dict)
    size = len(s)
    can_break = [False] * (size + 1)
    can_break[0] = True
    for p1 in range(1, size + 1):
        for p2 in range(p1):
            if can_break[p2] and s[p2:p1] in dict_set:
                can_break[p1] = True
                break
    return can_break[size]
'''

T14_WORD_BREAK_LRU_CACHE = '''
from functools import lru_cache

def __FUNC__(s, word_dict):
    # Word break top-down with lru_cache
    words = set(word_dict)
    @lru_cache(maxsize=None)
    def dfs(index):
        if index == len(s):
            return True
        for next_idx in range(index + 1, len(s) + 1):
            if s[index:next_idx] in words and dfs(next_idx):
                return True
        return False
    return dfs(0)
'''

TEMPLATES = [
    T1_MEMOIZED_1D, T2_TABULATED_1D, T3_2D_LCS, T4_2D_KNAPSACK, T5_LIS_1D, T6_HOUSE_ROBBER_1D,
    T7_WORD_BREAK_CLASSIC, T8_WORD_BREAK_WORD_BOUNDED, T9_WORD_BREAK_MEMO_RECURSION,
    T10_WORD_BREAK_ALT_NAMES, T11_WORD_BREAK_FORWARD_LOOP, T12_WORD_BREAK_MAX_LEN_BOUND,
    T13_WORD_BREAK_SUBPROBLEM_LIST, T14_WORD_BREAK_LRU_CACHE,
]