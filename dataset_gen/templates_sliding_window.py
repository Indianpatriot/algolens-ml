T1_FIXED_WINDOW_MAX_SUM = '''
def __FUNC__(__ARR__, __K__):
    # fixed-size sliding window, max sum of any window of size k
    __WINDOW_SUM__ = sum(__ARR__[:__K__])
    __MAXV__ = __WINDOW_SUM__
    for __I__ in range(__K__, len(__ARR__)):
        __WINDOW_SUM__ += __ARR__[__I__] - __ARR__[__I__ - __K__]
        __MAXV__ = max(__MAXV__, __WINDOW_SUM__)
    return __MAXV__
'''

T2_LONGEST_NO_REPEAT = '''
def __FUNC__(__ARR__):
    # variable-size sliding window, longest substring without repeats
    __SEEN__ = {}
    __LEFT__ = 0
    __MAXV__ = 0
    for __RIGHT__ in range(len(__ARR__)):
        __CH__ = __ARR__[__RIGHT__]
        if __CH__ in __SEEN__ and __SEEN__[__CH__] >= __LEFT__:
            __LEFT__ = __SEEN__[__CH__] + 1
        __SEEN__[__CH__] = __RIGHT__
        __MAXV__ = max(__MAXV__, __RIGHT__ - __LEFT__ + 1)
    return __MAXV__
'''

T3_MIN_WINDOW_SUBSTRING = '''
def __FUNC__(__ARR__, __TARGET__):
    # variable-size shrinking window, minimum window containing target chars
    __COUNT__ = {}
    for __CH__ in __TARGET__:
        __COUNT__[__CH__] = __COUNT__.get(__CH__, 0) + 1
    __NEED__ = len(__COUNT__)
    __HAVE__ = 0
    __LEFT__ = 0
    __BEST__ = (float('inf'), 0, 0)
    __WIN__ = {}
    for __RIGHT__ in range(len(__ARR__)):
        __CH__ = __ARR__[__RIGHT__]
        __WIN__[__CH__] = __WIN__.get(__CH__, 0) + 1
        if __CH__ in __COUNT__ and __WIN__[__CH__] == __COUNT__[__CH__]:
            __HAVE__ += 1
        while __HAVE__ == __NEED__:
            if __RIGHT__ - __LEFT__ + 1 < __BEST__[0]:
                __BEST__ = (__RIGHT__ - __LEFT__ + 1, __LEFT__, __RIGHT__)
            __LCHAR__ = __ARR__[__LEFT__]
            __WIN__[__LCHAR__] -= 1
            if __LCHAR__ in __COUNT__ and __WIN__[__LCHAR__] < __COUNT__[__LCHAR__]:
                __HAVE__ -= 1
            __LEFT__ += 1
    return "" if __BEST__[0] == float('inf') else __ARR__[__BEST__[1]:__BEST__[2] + 1]
'''

T4_LONGEST_SUBARRAY_SUM_LEQ_K = '''
def __FUNC__(__ARR__, __TARGET__):
    # variable-size sliding window, longest subarray with sum <= target
    __LEFT__ = 0
    __WINDOW_SUM__ = 0
    __MAXV__ = 0
    for __RIGHT__ in range(len(__ARR__)):
        __WINDOW_SUM__ += __ARR__[__RIGHT__]
        while __WINDOW_SUM__ > __TARGET__ and __LEFT__ <= __RIGHT__:
            __WINDOW_SUM__ -= __ARR__[__LEFT__]
            __LEFT__ += 1
        __MAXV__ = max(__MAXV__, __RIGHT__ - __LEFT__ + 1)
    return __MAXV__
'''

TEMPLATES = [T1_FIXED_WINDOW_MAX_SUM, T2_LONGEST_NO_REPEAT, T3_MIN_WINDOW_SUBSTRING, T4_LONGEST_SUBARRAY_SUM_LEQ_K]