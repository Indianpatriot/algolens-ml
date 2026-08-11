T1_CLASSIC = '''
def __FUNC__(__ARR__, __TARGET__):
    # classic binary search over a sorted array
    __LEFT__ = 0
    __RIGHT__ = len(__ARR__) - 1
    while __LEFT__ <= __RIGHT__:
        __MID__ = (__LEFT__ + __RIGHT__) // 2
        if __ARR__[__MID__] == __TARGET__:
            return __MID__
        elif __ARR__[__MID__] < __TARGET__:
            __LEFT__ = __MID__ + 1
        else:
            __RIGHT__ = __MID__ - 1
    return -1
'''

T2_ROTATED_ARRAY = '''
def __FUNC__(__ARR__, __TARGET__):
    # binary search in a rotated sorted array
    __LEFT__ = 0
    __RIGHT__ = len(__ARR__) - 1
    while __LEFT__ <= __RIGHT__:
        __MID__ = (__LEFT__ + __RIGHT__) // 2
        if __ARR__[__MID__] == __TARGET__:
            return __MID__
        if __ARR__[__LEFT__] <= __ARR__[__MID__]:
            if __ARR__[__LEFT__] <= __TARGET__ < __ARR__[__MID__]:
                __RIGHT__ = __MID__ - 1
            else:
                __LEFT__ = __MID__ + 1
        else:
            if __ARR__[__MID__] < __TARGET__ <= __ARR__[__RIGHT__]:
                __LEFT__ = __MID__ + 1
            else:
                __RIGHT__ = __MID__ - 1
    return -1
'''

T3_FIRST_LAST_OCCURRENCE = '''
def __FUNC__(__ARR__, __TARGET__):
    # binary search for the first and last occurrence of target
    def __HELPER__(__FIND_FIRST__):
        __LEFT__ = 0
        __RIGHT__ = len(__ARR__) - 1
        __RESULT__ = -1
        while __LEFT__ <= __RIGHT__:
            __MID__ = (__LEFT__ + __RIGHT__) // 2
            if __ARR__[__MID__] == __TARGET__:
                __RESULT__ = __MID__
                if __FIND_FIRST__:
                    __RIGHT__ = __MID__ - 1
                else:
                    __LEFT__ = __MID__ + 1
            elif __ARR__[__MID__] < __TARGET__:
                __LEFT__ = __MID__ + 1
            else:
                __RIGHT__ = __MID__ - 1
        return __RESULT__

    return [__HELPER__(True), __HELPER__(False)]
'''

T4_SEARCH_ON_ANSWER = '''
def __FUNC__(__TARGET__):
    # binary search on the answer space (integer sqrt)
    __LEFT__ = 0
    __RIGHT__ = __TARGET__
    __RESULT__ = 0
    while __LEFT__ <= __RIGHT__:
        __MID__ = (__LEFT__ + __RIGHT__) // 2
        if __MID__ * __MID__ <= __TARGET__:
            __RESULT__ = __MID__
            __LEFT__ = __MID__ + 1
        else:
            __RIGHT__ = __MID__ - 1
    return __RESULT__
'''

TEMPLATES = [T1_CLASSIC, T2_ROTATED_ARRAY, T3_FIRST_LAST_OCCURRENCE, T4_SEARCH_ON_ANSWER]