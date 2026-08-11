T1_BUBBLE_SORT = '''
def __FUNC__(__ARR__):
    # bubble sort - repeated adjacent swaps, early exit if sorted
    __N__ = len(__ARR__)
    for __I__ in range(__N__):
        __SWAPPED__ = False
        for __J__ in range(0, __N__ - __I__ - 1):
            if __ARR__[__J__] > __ARR__[__J__ + 1]:
                __ARR__[__J__], __ARR__[__J__ + 1] = __ARR__[__J__ + 1], __ARR__[__J__]
                __SWAPPED__ = True
        if not __SWAPPED__:
            break
    return __ARR__
'''

T2_INSERTION_SORT = '''
def __FUNC__(__ARR__):
    # insertion sort - shift elements right while building a sorted prefix
    for __I__ in range(1, len(__ARR__)):
        __KEY__ = __ARR__[__I__]
        __J__ = __I__ - 1
        while __J__ >= 0 and __ARR__[__J__] > __KEY__:
            __ARR__[__J__ + 1] = __ARR__[__J__]
            __J__ -= 1
        __ARR__[__J__ + 1] = __KEY__
    return __ARR__
'''

T3_SELECTION_SORT = '''
def __FUNC__(__ARR__):
    # selection sort - find min of the unsorted suffix, swap into place
    __N__ = len(__ARR__)
    for __I__ in range(__N__):
        __MIN_IDX__ = __I__
        for __J__ in range(__I__ + 1, __N__):
            if __ARR__[__J__] < __ARR__[__MIN_IDX__]:
                __MIN_IDX__ = __J__
        __ARR__[__I__], __ARR__[__MIN_IDX__] = __ARR__[__MIN_IDX__], __ARR__[__I__]
    return __ARR__
'''

T4_SHELL_SORT = '''
def __FUNC__(__ARR__):
    # shell sort - insertion sort with a shrinking gap sequence
    __N__ = len(__ARR__)
    __GAP__ = __N__ // 2
    while __GAP__ > 0:
        for __I__ in range(__GAP__, __N__):
            __TMP__ = __ARR__[__I__]
            __J__ = __I__
            while __J__ >= __GAP__ and __ARR__[__J__ - __GAP__] > __TMP__:
                __ARR__[__J__] = __ARR__[__J__ - __GAP__]
                __J__ -= __GAP__
            __ARR__[__J__] = __TMP__
        __GAP__ //= 2
    return __ARR__
'''

T5_MERGE_SORT = '''
def __FUNC__(__ARR__):
    # classic merge sort, divide and conquer with an auxiliary merge step
    if len(__ARR__) <= 1:
        return __ARR__
    __MID__ = len(__ARR__) // 2
    __LEFT__ = __FUNC__(__ARR__[:__MID__])
    __RIGHT__ = __FUNC__(__ARR__[__MID__:])
    __RESULT__ = []
    __I__ = __J__ = 0
    while __I__ < len(__LEFT__) and __J__ < len(__RIGHT__):
        if __LEFT__[__I__] <= __RIGHT__[__J__]:
            __RESULT__.append(__LEFT__[__I__])
            __I__ += 1
        else:
            __RESULT__.append(__RIGHT__[__J__])
            __J__ += 1
    __RESULT__.extend(__LEFT__[__I__:])
    __RESULT__.extend(__RIGHT__[__J__:])
    return __RESULT__
'''

T6_QUICK_SORT = '''
def __FUNC__(__ARR__, __START__=0, __END__=None):
    # in-place quick sort using Lomuto partition scheme
    if __END__ is None:
        __END__ = len(__ARR__) - 1
    if __START__ < __END__:
        __PIVOT__ = __ARR__[__END__]
        __I__ = __START__ - 1
        for __J__ in range(__START__, __END__):
            if __ARR__[__J__] <= __PIVOT__:
                __I__ += 1
                __ARR__[__I__], __ARR__[__J__] = __ARR__[__J__], __ARR__[__I__]
        __ARR__[__I__ + 1], __ARR__[__END__] = __ARR__[__END__], __ARR__[__I__ + 1]
        __PIVOT_IDX__ = __I__ + 1
        __FUNC__(__ARR__, __START__, __PIVOT_IDX__ - 1)
        __FUNC__(__ARR__, __PIVOT_IDX__ + 1, __END__)
    return __ARR__
'''

T7_HEAP_SORT = '''
def __FUNC__(__ARR__):
    # heap sort - build a max heap, then repeatedly extract the max
    __N__ = len(__ARR__)

    def __HELPER__(__HEAP_SIZE__, __I__):
        __LARGEST__ = __I__
        __L__ = 2 * __I__ + 1
        __R__ = 2 * __I__ + 2
        if __L__ < __HEAP_SIZE__ and __ARR__[__L__] > __ARR__[__LARGEST__]:
            __LARGEST__ = __L__
        if __R__ < __HEAP_SIZE__ and __ARR__[__R__] > __ARR__[__LARGEST__]:
            __LARGEST__ = __R__
        if __LARGEST__ != __I__:
            __ARR__[__I__], __ARR__[__LARGEST__] = __ARR__[__LARGEST__], __ARR__[__I__]
            __HELPER__(__HEAP_SIZE__, __LARGEST__)

    for __I__ in range(__N__ // 2 - 1, -1, -1):
        __HELPER__(__N__, __I__)
    for __I__ in range(__N__ - 1, 0, -1):
        __ARR__[0], __ARR__[__I__] = __ARR__[__I__], __ARR__[0]
        __HELPER__(__I__, 0)
    return __ARR__
'''

T8_TIMSORT = '''
def __FUNC__(__ARR__):
    # simplified timsort - insertion-sort small runs, then iterative bottom-up merge
    __N__ = len(__ARR__)
    __RUN__ = 8

    def insertion_pass(left, right):
        for i in range(left + 1, right + 1):
            key = __ARR__[i]
            j = i - 1
            while j >= left and __ARR__[j] > key:
                __ARR__[j + 1] = __ARR__[j]
                j -= 1
            __ARR__[j + 1] = key

    def merge_pass(left, mid, right):
        left_part = __ARR__[left:mid + 1]
        right_part = __ARR__[mid + 1:right + 1]
        i = j = 0
        k = left
        while i < len(left_part) and j < len(right_part):
            if left_part[i] <= right_part[j]:
                __ARR__[k] = left_part[i]
                i += 1
            else:
                __ARR__[k] = right_part[j]
                j += 1
            k += 1
        while i < len(left_part):
            __ARR__[k] = left_part[i]
            i += 1
            k += 1
        while j < len(right_part):
            __ARR__[k] = right_part[j]
            j += 1
            k += 1

    for start in range(0, __N__, __RUN__):
        insertion_pass(start, min(start + __RUN__ - 1, __N__ - 1))

    __SIZE__ = __RUN__
    while __SIZE__ < __N__:
        for left in range(0, __N__, 2 * __SIZE__):
            mid = min(left + __SIZE__ - 1, __N__ - 1)
            right = min(left + 2 * __SIZE__ - 1, __N__ - 1)
            if mid < right:
                merge_pass(left, mid, right)
        __SIZE__ *= 2
    return __ARR__
'''

T9_COUNTING_SORT = '''
def __FUNC__(__ARR__):
    # counting sort - frequency array + prefix sum, non-negative integers only
    if not __ARR__:
        return __ARR__
    __MAXV__ = max(__ARR__)
    __MINV__ = min(__ARR__)
    __SPAN__ = __MAXV__ - __MINV__ + 1
    __COUNT__ = [0] * __SPAN__
    for __NUM__ in __ARR__:
        __COUNT__[__NUM__ - __MINV__] += 1
    for __I__ in range(1, __SPAN__):
        __COUNT__[__I__] += __COUNT__[__I__ - 1]
    __RESULT__ = [0] * len(__ARR__)
    for __NUM__ in reversed(__ARR__):
        __COUNT__[__NUM__ - __MINV__] -= 1
        __RESULT__[__COUNT__[__NUM__ - __MINV__]] = __NUM__
    return __RESULT__
'''

T10_RADIX_SORT = '''
def _radix_counting_pass(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        digit = (arr[i] // exp) % 10
        count[digit] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        count[digit] -= 1
        output[count[digit]] = arr[i]
    return output

def __FUNC__(__ARR__):
    # LSD radix sort - repeated counting sort passes, one per decimal digit
    if not __ARR__:
        return __ARR__
    __MAXV__ = max(__ARR__)
    __EXP__ = 1
    while __MAXV__ // __EXP__ > 0:
        __ARR__ = _radix_counting_pass(__ARR__, __EXP__)
        __EXP__ *= 10
    return __ARR__
'''

T11_BUCKET_SORT = '''
def __FUNC__(__ARR__):
    # bucket sort - distribute floats in [0, 1) into n buckets, sort each, concatenate
    __N__ = len(__ARR__)
    if __N__ == 0:
        return __ARR__
    __BUCKETS__ = [[] for _ in range(__N__)]
    for __NUM__ in __ARR__:
        __IDX__ = int(__NUM__ * __N__)
        if __IDX__ == __N__:
            __IDX__ = __N__ - 1
        __BUCKETS__[__IDX__].append(__NUM__)
    for __BUCKET__ in __BUCKETS__:
        __BUCKET__.sort()
    __RESULT__ = []
    for __BUCKET__ in __BUCKETS__:
        __RESULT__.extend(__BUCKET__)
    return __RESULT__
'''

T12_TREE_SORT = '''
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def _bst_insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = _bst_insert(root.left, val)
    else:
        root.right = _bst_insert(root.right, val)
    return root

def _inorder(root, out):
    if root:
        _inorder(root.left, out)
        out.append(root.val)
        _inorder(root.right, out)

def __FUNC__(__ARR__):
    # tree sort - insert into a BST, then inorder traversal yields sorted order
    __ROOT__ = None
    for __NUM__ in __ARR__:
        __ROOT__ = _bst_insert(__ROOT__, __NUM__)
    __RESULT__ = []
    _inorder(__ROOT__, __RESULT__)
    return __RESULT__
'''

T13_CYCLE_SORT = '''
def __FUNC__(__ARR__):
    # cycle sort - minimizes writes, places each element on its correct cycle
    __N__ = len(__ARR__)
    for __START__ in range(__N__ - 1):
        __ITEM__ = __ARR__[__START__]
        __POS__ = __START__
        for __I__ in range(__START__ + 1, __N__):
            if __ARR__[__I__] < __ITEM__:
                __POS__ += 1
        if __POS__ == __START__:
            continue
        while __ITEM__ == __ARR__[__POS__]:
            __POS__ += 1
        __ARR__[__POS__], __ITEM__ = __ITEM__, __ARR__[__POS__]
        while __POS__ != __START__:
            __POS__ = __START__
            for __I__ in range(__START__ + 1, __N__):
                if __ARR__[__I__] < __ITEM__:
                    __POS__ += 1
            while __ITEM__ == __ARR__[__POS__]:
                __POS__ += 1
            __ARR__[__POS__], __ITEM__ = __ITEM__, __ARR__[__POS__]
    return __ARR__
'''

T14_PIGEONHOLE_SORT = '''
def __FUNC__(__ARR__):
    # pigeonhole sort - one hole per possible value, append then flatten
    if not __ARR__:
        return __ARR__
    __MINV__ = min(__ARR__)
    __MAXV__ = max(__ARR__)
    __HOLES__ = [[] for _ in range(__MAXV__ - __MINV__ + 1)]
    for __NUM__ in __ARR__:
        __HOLES__[__NUM__ - __MINV__].append(__NUM__)
    __RESULT__ = []
    for __HOLE__ in __HOLES__:
        __RESULT__.extend(__HOLE__)
    return __RESULT__
'''

T15_BITONIC_SORT = '''
def _compare_and_swap(arr, i, j, ascending):
    if (arr[i] > arr[j]) == ascending:
        arr[i], arr[j] = arr[j], arr[i]

def _bitonic_merge(arr, low, length, ascending):
    if length > 1:
        mid = length // 2
        for i in range(low, low + mid):
            _compare_and_swap(arr, i, i + mid, ascending)
        _bitonic_merge(arr, low, mid, ascending)
        _bitonic_merge(arr, low + mid, mid, ascending)

def _bitonic_sort_recursive(arr, low, length, ascending):
    if length > 1:
        mid = length // 2
        _bitonic_sort_recursive(arr, low, mid, True)
        _bitonic_sort_recursive(arr, low + mid, mid, False)
        _bitonic_merge(arr, low, length, ascending)

def __FUNC__(__ARR__):
    # bitonic sort - comparator network, requires len(arr) to be a power of 2
    _bitonic_sort_recursive(__ARR__, 0, len(__ARR__), True)
    return __ARR__
'''

T16_GNOME_SORT = '''
def __FUNC__(__ARR__):
    # gnome sort - single index walks forward, steps back on an inversion
    __N__ = len(__ARR__)
    __INDEX__ = 0
    while __INDEX__ < __N__:
        if __INDEX__ == 0:
            __INDEX__ += 1
        elif __ARR__[__INDEX__] >= __ARR__[__INDEX__ - 1]:
            __INDEX__ += 1
        else:
            __ARR__[__INDEX__], __ARR__[__INDEX__ - 1] = __ARR__[__INDEX__ - 1], __ARR__[__INDEX__]
            __INDEX__ -= 1
    return __ARR__
'''

T17_COMB_SORT = '''
def __FUNC__(__ARR__):
    # comb sort - bubble sort with a shrinking gap instead of gap=1
    __N__ = len(__ARR__)
    __GAP__ = __N__
    __SHRINK__ = 1.3
    __SWAPPED__ = True
    while __GAP__ > 1 or __SWAPPED__:
        __GAP__ = int(__GAP__ / __SHRINK__)
        if __GAP__ < 1:
            __GAP__ = 1
        __SWAPPED__ = False
        for __I__ in range(__N__ - __GAP__):
            if __ARR__[__I__] > __ARR__[__I__ + __GAP__]:
                __ARR__[__I__], __ARR__[__I__ + __GAP__] = __ARR__[__I__ + __GAP__], __ARR__[__I__]
                __SWAPPED__ = True
    return __ARR__
'''

T18_PANCAKE_SORT = '''
def _flip(arr, k):
    arr[:k + 1] = arr[:k + 1][::-1]

def _find_max_index(arr, n):
    max_idx = 0
    for i in range(1, n):
        if arr[i] > arr[max_idx]:
            max_idx = i
    return max_idx

def __FUNC__(__ARR__):
    # pancake sort - repeatedly flip the prefix to move the max into place
    __N__ = len(__ARR__)
    for __SIZE__ in range(__N__, 1, -1):
        __MAX_IDX__ = _find_max_index(__ARR__, __SIZE__)
        if __MAX_IDX__ != __SIZE__ - 1:
            _flip(__ARR__, __MAX_IDX__)
            _flip(__ARR__, __SIZE__ - 1)
    return __ARR__
'''

T19_INTROSORT = '''
import math

def _insertion_range(arr, start, end):
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def _heapify_range(arr, n, i, offset):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[offset + left] > arr[offset + largest]:
        largest = left
    if right < n and arr[offset + right] > arr[offset + largest]:
        largest = right
    if largest != i:
        arr[offset + i], arr[offset + largest] = arr[offset + largest], arr[offset + i]
        _heapify_range(arr, n, largest, offset)

def _heap_sort_range(arr, start, end):
    n = end - start + 1
    for i in range(n // 2 - 1, -1, -1):
        _heapify_range(arr, n, i, start)
    for i in range(n - 1, 0, -1):
        arr[start], arr[start + i] = arr[start + i], arr[start]
        _heapify_range(arr, i, 0, start)

def _partition_range(arr, start, end):
    pivot = arr[end]
    i = start - 1
    for j in range(start, end):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[end] = arr[end], arr[i + 1]
    return i + 1

def _introsort_helper(arr, start, end, depth_limit):
    size = end - start + 1
    if size < 16:
        _insertion_range(arr, start, end)
        return
    if depth_limit == 0:
        _heap_sort_range(arr, start, end)
        return
    pivot_idx = _partition_range(arr, start, end)
    _introsort_helper(arr, start, pivot_idx - 1, depth_limit - 1)
    _introsort_helper(arr, pivot_idx + 1, end, depth_limit - 1)

def __FUNC__(__ARR__):
    # introsort - quicksort that falls back to heapsort past a recursion-depth limit
    __N__ = len(__ARR__)
    __DEPTH_LIMIT__ = 2 * int(math.log2(__N__)) if __N__ > 1 else 1
    _introsort_helper(__ARR__, 0, __N__ - 1, __DEPTH_LIMIT__)
    return __ARR__
'''

T20_SMOOTHSORT = '''
def _leonardo(k):
    if k < 2:
        return 1
    a, b = 1, 1
    for _ in range(k - 1):
        a, b = b, a + b + 1
    return b

def _sift(arr, root, order):
    while order > 1:
        right_child = root - 1
        left_child = root - 1 - _leonardo(order - 2)
        if arr[left_child] >= arr[right_child]:
            bigger_child = left_child
            next_order = order - 1
        else:
            bigger_child = right_child
            next_order = order - 2
        if arr[bigger_child] <= arr[root]:
            break
        arr[root], arr[bigger_child] = arr[bigger_child], arr[root]
        root = bigger_child
        order = next_order

def _heap_root_positions(heap_orders):
    positions = []
    pos = -1
    for order in heap_orders:
        pos += _leonardo(order)
        positions.append(pos)
    return positions

def __FUNC__(__ARR__):
    # simplified smoothsort - a forest of Leonardo heaps. Each extraction step
    # explicitly selects the max among current heap roots and swaps it into
    # place, rather than replicating Dijkstra's O(1)-amortized trinkle - still
    # uses the real Leonardo-heap sift/order machinery, just a simpler (and
    # asymptotically slightly less optimal, but definitely correct) selection
    # step for restoring cross-heap ordering.
    __N__ = len(__ARR__)
    __HEAP_ORDERS__ = []
    for __END__ in range(__N__):
        if len(__HEAP_ORDERS__) >= 2 and __HEAP_ORDERS__[-2] == __HEAP_ORDERS__[-1] + 1:
            __HEAP_ORDERS__[-2] += 1
            __HEAP_ORDERS__.pop()
        elif __HEAP_ORDERS__ and __HEAP_ORDERS__[-1] == 1:
            __HEAP_ORDERS__.append(0)
        else:
            __HEAP_ORDERS__.append(1)
        _sift(__ARR__, __END__, __HEAP_ORDERS__[-1])

    for __END__ in range(__N__ - 1, -1, -1):
        if not __HEAP_ORDERS__:
            break
        __POSITIONS__ = _heap_root_positions(__HEAP_ORDERS__)
        __MAX_K__ = max(range(len(__POSITIONS__)), key=lambda i: __ARR__[__POSITIONS__[i]])
        if __MAX_K__ != len(__POSITIONS__) - 1:
            __ARR__[__POSITIONS__[__MAX_K__]], __ARR__[__POSITIONS__[-1]] = __ARR__[__POSITIONS__[-1]], __ARR__[__POSITIONS__[__MAX_K__]]
            _sift(__ARR__, __POSITIONS__[__MAX_K__], __HEAP_ORDERS__[__MAX_K__])
        __TOP_ORDER__ = __HEAP_ORDERS__[-1]
        if __TOP_ORDER__ <= 1:
            __HEAP_ORDERS__.pop()
        else:
            __HEAP_ORDERS__.pop()
            __HEAP_ORDERS__.append(__TOP_ORDER__ - 1)
            __HEAP_ORDERS__.append(__TOP_ORDER__ - 2)
    return __ARR__
'''

T21_BOGO_SORT = '''
import random

def _is_sorted(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def __FUNC__(__ARR__):
    # bogo sort - random shuffle until sorted, capped to avoid a runaway loop
    __ATTEMPTS__ = 0
    while not _is_sorted(__ARR__) and __ATTEMPTS__ < 2000:
        random.shuffle(__ARR__)
        __ATTEMPTS__ += 1
    return __ARR__
'''

TEMPLATES = [
    T1_BUBBLE_SORT, T2_INSERTION_SORT, T3_SELECTION_SORT, T4_SHELL_SORT,
    T5_MERGE_SORT, T6_QUICK_SORT, T7_HEAP_SORT, T8_TIMSORT,
    T9_COUNTING_SORT, T10_RADIX_SORT, T11_BUCKET_SORT, T12_TREE_SORT,
    T13_CYCLE_SORT, T14_PIGEONHOLE_SORT, T15_BITONIC_SORT, T16_GNOME_SORT,
    T17_COMB_SORT, T18_PANCAKE_SORT, T19_INTROSORT, T20_SMOOTHSORT,
    T21_BOGO_SORT,
]

TEMPLATE_NAMES = [
    "Bubble Sort", "Insertion Sort", "Selection Sort", "Shell Sort",
    "Merge Sort", "Quick Sort", "Heap Sort", "Timsort",
    "Counting Sort", "Radix Sort", "Bucket Sort", "Tree Sort",
    "Cycle Sort", "Pigeonhole Sort", "Bitonic Sort", "Gnome Sort",
    "Comb Sort", "Pancake Sort", "Introsort", "Smoothsort",
    "Bogo Sort",
]

# Sub-labels used for training instead of one flat "Sorting" bucket.
# Aligned index-for-index with TEMPLATES / TEMPLATE_NAMES above.
SUB_LABELS = [
    "In-Place Nested-Loop Swap Sort",   # Bubble Sort
    "Shift-Based Sort",                  # Insertion Sort
    "In-Place Nested-Loop Swap Sort",   # Selection Sort
    "Shift-Based Sort",                  # Shell Sort
    "Recursive Non-Swap Sort",           # Merge Sort
    "Recursive Swap-Based Sort",         # Quick Sort
    "Heap/Complex Structural Sort",      # Heap Sort
    "Heap/Complex Structural Sort",      # Timsort
    "Distribution Sort",                 # Counting Sort
    "Distribution Sort",                 # Radix Sort
    "Distribution Sort",                 # Bucket Sort
    "Recursive Non-Swap Sort",           # Tree Sort
    "In-Place Nested-Loop Swap Sort",   # Cycle Sort
    "Distribution Sort",                 # Pigeonhole Sort
    "Recursive Swap-Based Sort",         # Bitonic Sort
    "Single-Pass Loop Sort",             # Gnome Sort
    "In-Place Nested-Loop Swap Sort",   # Comb Sort
    "Shift-Based Sort",                  # Pancake Sort
    "Recursive Swap-Based Sort",         # Introsort
    "Heap/Complex Structural Sort",      # Smoothsort
    "Single-Pass Loop Sort",             # Bogo Sort
]