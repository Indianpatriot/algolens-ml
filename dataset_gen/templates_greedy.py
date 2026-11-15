"""
Templates for Greedy Algorithms (Interval Scheduling, Non-overlapping intervals, Merge intervals, Meeting rooms).
Generates diverse code samples for training the classifier.
"""

T1_INTERVAL_SCHEDULING = '''
def __FUNC__(intervals: list[list[int]]) -> list[list[int]]:
    # Greedy interval scheduling / activity selection by earliest finish time
    intervals.sort(key=lambda x: x[1])
    selected = []
    last_end_time = float('-inf')

    for start, end in intervals:
        if start >= last_end_time:
            selected.append([start, end])
            last_end_time = end

    return selected
'''

T2_NON_OVERLAPPING_INTERVALS = '''
def __FUNC__(intervals: list[list[int]]) -> int:
    # LeetCode 435: Erase overlap intervals (earliest finish time greedy)
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 0
    prev_end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] < prev_end:
            count += 1
        else:
            prev_end = intervals[i][1]

    return count
'''

T3_MERGE_INTERVALS = '''
def __FUNC__(intervals: list[list[int]]) -> list[list[int]]:
    # Merge overlapping intervals after sorting by start time
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for i in range(1, len(intervals)):
        current = intervals[i]
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current[1])
        else:
            merged.append(current)

    return merged
'''

T4_MEETING_ROOMS = '''
def __FUNC__(intervals: list[list[int]]) -> bool:
    # Determine if person can attend all meetings (greedy overlap check)
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i + 1][0]:
            return False
    return True
'''

TEMPLATES = [
    T1_INTERVAL_SCHEDULING,
    T2_NON_OVERLAPPING_INTERVALS,
    T3_MERGE_INTERVALS,
    T4_MEETING_ROOMS,
]
