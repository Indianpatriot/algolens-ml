from __future__ import annotations

import json

from execution_tracer import trace_execution


SAMPLES = {
    "bubble_sort": """
def bubble_sort(nums):
    arr = nums[:]
    for i in range(len(arr)):
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

print(bubble_sort([5, 1, 4, 2]))
""",
    "bfs": """
from collections import deque

def bfs(graph, start):
    seen = set([start])
    order = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order

graph = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": [],
}
print(bfs(graph, "A"))
""",
    "binary_search": """
def binary_search(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

print(binary_search([1, 3, 5, 7, 9], 7))
""",
    "recursive_fibonacci": """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(5))
""",
}


def main() -> None:
    for name, code in SAMPLES.items():
        result = trace_execution(code)
        steps = result["steps"]
        print(f"\n=== {name} ===")
        print(f"error: {result['error']}")
        print(f"stdout: {result['final_output'].strip()!r}")
        print(f"step_count: {len(steps)}")
        for index, step in enumerate(steps[:12], start=1):
            compact = {
                "step": index,
                "event": step["event"],
                "line_number": step["line_number"],
                "line_text": step["line_text"].strip(),
                "call_depth": step["call_depth"],
                "locals": step["locals"],
            }
            print(json.dumps(compact, sort_keys=True))
        if len(steps) > 12:
            print(f"... {len(steps) - 12} more steps")


if __name__ == "__main__":
    main()
