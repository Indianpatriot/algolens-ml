T1_BASIC_PATH_COMPRESSION = '''
def make_set(__N__):
    return list(range(__N__))

def find(__PARENT__, __I__):
    # union-find with path compression
    if __PARENT__[__I__] != __I__:
        __PARENT__[__I__] = find(__PARENT__, __PARENT__[__I__])
    return __PARENT__[__I__]

def union(__PARENT__, __X__, __Y__):
    __ROOT_X__ = find(__PARENT__, __X__)
    __ROOT_Y__ = find(__PARENT__, __Y__)
    if __ROOT_X__ != __ROOT_Y__:
        __PARENT__[__ROOT_X__] = __ROOT_Y__
'''

T2_UNION_BY_RANK = '''
def __FUNC__(__N__, __ARR__):
    # union-find with union by rank
    __PARENT__ = list(range(__N__))
    __RANK__ = [0] * __N__

    def find(__X__):
        while __PARENT__[__X__] != __X__:
            __PARENT__[__X__] = __PARENT__[__PARENT__[__X__]]
            __X__ = __PARENT__[__X__]
        return __X__

    def union(__A__, __B__):
        __ROOT_A__ = find(__A__)
        __ROOT_B__ = find(__B__)
        if __ROOT_A__ == __ROOT_B__:
            return
        if __RANK__[__ROOT_A__] < __RANK__[__ROOT_B__]:
            __PARENT__[__ROOT_A__] = __ROOT_B__
        elif __RANK__[__ROOT_A__] > __RANK__[__ROOT_B__]:
            __PARENT__[__ROOT_B__] = __ROOT_A__
        else:
            __PARENT__[__ROOT_B__] = __ROOT_A__
            __RANK__[__ROOT_A__] += 1

    for __U__, __V__ in __ARR__:
        union(__U__, __V__)
    return __PARENT__
'''

T3_COUNT_COMPONENTS = '''
def __FUNC__(__N__, __ARR__):
    # count connected components using union-find
    __PARENT__ = list(range(__N__))

    def find(__X__):
        if __PARENT__[__X__] != __X__:
            __PARENT__[__X__] = find(__PARENT__[__X__])
        return __PARENT__[__X__]

    def union(__A__, __B__):
        __ROOT_A__, __ROOT_B__ = find(__A__), find(__B__)
        if __ROOT_A__ != __ROOT_B__:
            __PARENT__[__ROOT_A__] = __ROOT_B__

    for __U__, __V__ in __ARR__:
        union(__U__, __V__)
    return len(set(find(__X__) for __X__ in range(__N__)))
'''

T4_REDUNDANT_CONNECTION = '''
def __FUNC__(__ARR__):
    # detect the redundant edge that creates a cycle, via union-find
    __N__ = len(__ARR__)
    __PARENT__ = list(range(__N__ + 1))

    def find(__X__):
        if __PARENT__[__X__] != __X__:
            __PARENT__[__X__] = find(__PARENT__[__X__])
        return __PARENT__[__X__]

    for __U__, __V__ in __ARR__:
        __ROOT_U__ = find(__U__)
        __ROOT_V__ = find(__V__)
        if __ROOT_U__ == __ROOT_V__:
            return [__U__, __V__]
        __PARENT__[__ROOT_U__] = __ROOT_V__
    return []
'''

TEMPLATES = [T1_BASIC_PATH_COMPRESSION, T2_UNION_BY_RANK, T3_COUNT_COMPONENTS, T4_REDUNDANT_CONNECTION]