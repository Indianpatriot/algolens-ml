T1_NEXT_GREATER_ELEMENT = '''
def __FUNC__(__ARR__):
    # monotonic decreasing stack, next greater element
    __N__ = len(__ARR__)
    __RESULT__ = [-1] * __N__
    __STACK__ = []
    for __I__ in range(__N__):
        while __STACK__ and __ARR__[__STACK__[-1]] < __ARR__[__I__]:
            __TOP__ = __STACK__.pop()
            __RESULT__[__TOP__] = __ARR__[__I__]
        __STACK__.append(__I__)
    return __RESULT__
'''

T2_DAILY_TEMPERATURES = '''
def __FUNC__(__ARR__):
    # monotonic stack of indices, days until a warmer temperature
    __RESULT__ = [0] * len(__ARR__)
    __STACK__ = []
    for __I__, __CUR__ in enumerate(__ARR__):
        while __STACK__ and __ARR__[__STACK__[-1]] < __CUR__:
            __PREV_IDX__ = __STACK__.pop()
            __RESULT__[__PREV_IDX__] = __I__ - __PREV_IDX__
        __STACK__.append(__I__)
    return __RESULT__
'''

T3_LARGEST_RECTANGLE_HISTOGRAM = '''
def __FUNC__(__ARR__):
    # monotonic increasing stack, largest rectangle in a histogram
    __STACK__ = []
    __MAXV__ = 0
    __N__ = len(__ARR__)
    for __I__ in range(__N__ + 1):
        __CUR__ = __ARR__[__I__] if __I__ < __N__ else 0
        while __STACK__ and __ARR__[__STACK__[-1]] >= __CUR__:
            __HEIGHT__ = __ARR__[__STACK__.pop()]
            __WIDTH__ = __I__ if not __STACK__ else __I__ - __STACK__[-1] - 1
            __MAXV__ = max(__MAXV__, __HEIGHT__ * __WIDTH__)
        __STACK__.append(__I__)
    return __MAXV__
'''

T4_TRAPPING_RAIN_WATER = '''
def __FUNC__(__ARR__):
    # monotonic stack approach to trapping rain water
    __STACK__ = []
    __RESULT__ = 0
    for __I__, __CUR__ in enumerate(__ARR__):
        while __STACK__ and __ARR__[__STACK__[-1]] < __CUR__:
            __TOP__ = __STACK__.pop()
            if not __STACK__:
                break
            __LEFT_IDX__ = __STACK__[-1]
            __WIDTH__ = __I__ - __LEFT_IDX__ - 1
            __BOUNDED_HEIGHT__ = min(__ARR__[__LEFT_IDX__], __CUR__) - __ARR__[__TOP__]
            __RESULT__ += __WIDTH__ * __BOUNDED_HEIGHT__
        __STACK__.append(__I__)
    return __RESULT__
'''

TEMPLATES = [T1_NEXT_GREATER_ELEMENT, T2_DAILY_TEMPERATURES, T3_LARGEST_RECTANGLE_HISTOGRAM, T4_TRAPPING_RAIN_WATER]