"""
Linked List Operations templates - 4 structurally distinct sub-patterns:
  1. Insert at head (O(1), no traversal)
  2. Insert at position (bounded traversal + pointer relinking)
  3. Delete by value (traversal with a trailing 'prev' pointer)
  4. Reverse iteratively (full traversal, reversing .next at each step)

These are deliberately NOT tokenized with the __TOKEN__ naming-pool system
for head/node/data/prev/current - those names are so conventional in real
linked-list code that keeping them literal (rather than randomizing) is
more representative of how students actually write this, and matches how
the Fast/Slow-Pointer template already handles "slow"/"fast"/"head".

Only __FUNC__ (function name) and __TARGET__ (the value being
inserted/deleted/searched for) are tokenized, since those genuinely vary.
"""

TEMPLATES = [
    # 1. Insert at head - O(1), no traversal needed
    """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def __FUNC__(head, __TARGET__):
    new_node = Node(__TARGET__)
    new_node.next = head
    return new_node
""",

    # 2. Insert at a given position - bounded traversal, then relink
    """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def __FUNC__(head, __TARGET__, position):
    new_node = Node(__TARGET__)

    if position == 0:
        new_node.next = head
        return new_node

    current = head
    for _ in range(position - 1):
        if current is None:
            return head
        current = current.next

    if current is None:
        return head

    new_node.next = current.next
    current.next = new_node
    return head
""",

    # 3. Delete by value - traversal with a trailing 'prev' pointer
    """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def __FUNC__(head, __TARGET__):
    if head is None:
        return None

    if head.data == __TARGET__:
        return head.next

    prev = head
    current = head.next
    while current is not None:
        if current.data == __TARGET__:
            prev.next = current.next
            return head
        prev = current
        current = current.next

    return head
""",

    # 4. Reverse iteratively - full traversal, flipping .next each step
    """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def __FUNC__(head):
    prev = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
""",
]