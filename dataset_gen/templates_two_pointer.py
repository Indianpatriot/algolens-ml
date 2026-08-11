"""
Two Pointer templates - 4 structurally distinct sub-patterns:
  1. Opposite Ends (Converging)
  2. Same-Direction (Unidirectional slow/fast index over an array)
  3. Fast and Slow (Tortoise and Hare, linked list)
  4. Parallel Pointers (Two Arrays)

Each is written with __TOKEN__ placeholders per common.py's render().
Note: fast/slow/head/next are left as literal identifiers (not tokenized)
since they are conventional, invariant names for the tortoise-hare pattern,
matching how has_fast_slow_pointer_pattern / has_linked_list_pattern detect
`fast.next.next`-style access regardless of naming style.
"""

TEMPLATES = [
    # 1. Opposite Ends (Converging Pointers) - two-sum on a sorted array
    """
def __FUNC__(__ARR__, __TARGET__):
    # Two pointers start at opposite ends and move toward each other
    __LEFT__, __RIGHT__ = 0, len(__ARR__) - 1
    while __LEFT__ < __RIGHT__:
        __CUR__ = __ARR__[__LEFT__] + __ARR__[__RIGHT__]
        if __CUR__ == __TARGET__:
            return [__LEFT__, __RIGHT__]
        elif __CUR__ < __TARGET__:
            __LEFT__ += 1
        else:
            __RIGHT__ -= 1
    return [-1, -1]
""",

    # 2. Same-Direction (Unidirectional) - remove duplicates in place
    """
def __FUNC__(__ARR__):
    # One index (i) trails behind, only advancing when a new distinct
    # value is found; the other (j) scans ahead every iteration.
    if not __ARR__:
        return 0
    __I__ = 0
    for __J__ in range(1, len(__ARR__)):
        if __ARR__[__J__] != __ARR__[__I__]:
            __I__ += 1
            __ARR__[__I__] = __ARR__[__J__]
    return __I__ + 1
""",

    # 3. Fast and Slow (Tortoise and Hare) - linked list cycle detection
    """
def __FUNC__(head):
    # slow advances one step, fast advances two steps per iteration
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
""",

    # 4. Parallel Pointers (Two Arrays) - merge two sorted arrays
    """
def __FUNC__(__ARR__, arr2):
    # Two independent indices, each advancing over its own array
    __I__, __J__ = 0, 0
    __RESULT__ = []
    while __I__ < len(__ARR__) and __J__ < len(arr2):
        if __ARR__[__I__] <= arr2[__J__]:
            __RESULT__.append(__ARR__[__I__])
            __I__ += 1
        else:
            __RESULT__.append(arr2[__J__])
            __J__ += 1
    __RESULT__.extend(__ARR__[__I__:])
    __RESULT__.extend(arr2[__J__:])
    return __RESULT__
""",
]