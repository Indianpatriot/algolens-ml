"""
feature_extractor.py

Parses Python source code into an AST and extracts a fixed-length
numeric feature vector describing its structural properties.
These features are used downstream by a trained classifier to
predict which algorithm category the code implements, and by a
rule-based module to estimate time/space complexity.

Detection is primarily based on AST node types and call/import
structure. A few heuristics (e.g. "visited" in a variable name,
"parent"/"find"/"union" function names) are SECONDARY signals —
they carry real information in practice but are not purely
structural, and are documented as such below.
"""

import ast
from typing import Dict


FEATURE_NAMES = [
    "max_loop_depth",
    "num_loops",
    "has_recursion",
    "recursion_branch_count",
    "has_memoization",
    "uses_priority_queue",
    "uses_queue",
    "uses_stack",
    "uses_set",
    "uses_dict",
    "has_visited_like_var",
    "has_2d_list_indexing",
    "array_dimensionality",
    "has_swap_in_loop",
    "uses_sorted_or_sort_call",
    "has_two_pointer_pattern",
    "num_function_defs",
    "max_if_nesting_in_loop",
    "has_early_return_in_loop",
    "param_count",
    "uses_adjacency_structure",
    "uses_bit_manipulation",
    "has_linked_list_pattern",
    "has_ll_insert_pattern",
    "has_ll_delete_pattern",
    "has_ll_reverse_pattern",
    "has_tree_pattern",
    "has_prefix_pattern",
    "has_backtracking_undo_pattern",
    "has_union_find_pattern",
    "has_fast_slow_pointer_pattern",
    "has_sliding_window_pattern",
    "has_monotonic_stack_pattern",
    "has_topological_sort_pattern",
    "has_matrix_transpose_pattern",
    "uses_modular_math",
    "has_same_direction_pointer_pattern",
    "has_parallel_two_pointer_pattern",
    "has_midpoint_calculation",
    # Algorithm-specific structural AST signals for sorting & fine-grained classification
    "has_dual_recursion",
    "has_pivot_partition",
    "has_lomuto_hoare_indexing",
    "has_split_slicing",
    "has_auxiliary_merging",
    "has_key_assignment",
    "has_element_shifting",
    "has_gap_reduction",
    "has_heapify_pattern",
    "has_heap_child_math",
    "has_frequency_array",
    "has_radix_exp_scaling",
]

_BITWISE_OPS = (ast.BitXor, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift)


class _StructuralVisitor(ast.NodeVisitor):
    def __init__(self):
        # loops
        self.max_loop_depth = 0
        self.num_loops = 0
        self._current_loop_depth = 0
        self._inside_loop_depth_for_if = 0

        # functions / recursion
        self.function_names = set()
        self.recursive_calls = {}
        self.current_function_stack = []
        self.num_function_defs = 0
        self.param_count = 0

        # imports / libs
        self.imports = set()
        self.uses_priority_queue = False
        self.uses_queue = False
        self.uses_stack_lib = False

        # data structure variables
        self.dict_vars = set()
        self.set_vars = set()
        self.list_vars = set()
        self.memo_like_vars = set()  # tighter subset: named like a cache, or a default-arg cache

        # secondary naming heuristics
        self.has_visited_like_var = False
        self.uses_adjacency_structure = False

        # indexing / arrays
        self.has_2d_list_indexing = False
        self.max_subscript_depth = 0

        # misc structural
        self.has_swap_in_loop = False
        self.uses_sorted_or_sort_call = False
        self.has_two_pointer_pattern = False
        self.max_if_nesting_in_loop = 0
        self._current_if_nesting = 0
        self.has_early_return_in_loop = False

        # batch 2 features
        self.uses_bit_manipulation = False
        self.has_linked_list_pattern = False
        self.has_ll_insert_pattern = False
        self.has_ll_delete_pattern = False
        self.has_ll_reverse_pattern = False
        self.has_tree_pattern = False
        self.has_prefix_pattern = False

        # batch 3 features
        self.has_backtracking_undo_pattern = False
        self.has_union_find_pattern = False
        self.has_fast_slow_pointer_pattern = False
        self.has_sliding_window_pattern = False
        self.has_monotonic_stack_pattern = False
        self.has_topological_sort_pattern = False
        self.has_matrix_transpose_pattern = False
        self.uses_modular_math = False

        # Two-pointer sub-variant signals
        self.has_same_direction_pointer_pattern = False
        self.has_parallel_two_pointer_pattern = False
        self.has_midpoint_calculation = False

        # Algorithm-specific structural AST signals for sorting
        self.has_dual_recursion = False
        self.has_pivot_partition = False
        self.has_lomuto_hoare_indexing = False
        self.has_split_slicing = False
        self.has_auxiliary_merging = False
        self.has_key_assignment = False
        self.has_element_shifting = False
        self.has_gap_reduction = False
        self.has_heapify_pattern = False
        self.has_heap_child_math = False
        self.has_frequency_array = False
        self.has_radix_exp_scaling = False

        self._union_find_scanned = False

    # ================= Module-level pre-scan =================
    def visit_Module(self, node):
        self._scan_union_find(node)
        self._scan_topological_sort(node)
        self._scan_linked_list_operation(node)
        self._scan_sorting_structures(node)
        self.generic_visit(node)

    def _scan_sorting_structures(self, tree):
        """Scans module AST for sorting structural keywords and patterns."""
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                lname = n.name.lower()
                if any(k in lname for k in ("heapify", "sift", "percolate", "heapsort", "heap_sort")):
                    self.has_heapify_pattern = True
                if any(k in lname for k in ("radix", "count_sort", "counting_pass")):
                    self.has_radix_exp_scaling = True
                if "partition" in lname:
                    self.has_pivot_partition = True
            elif isinstance(n, ast.Name):
                lname = n.id.lower()
                if "pivot" in lname:
                    self.has_pivot_partition = True
                if "gap" in lname or "shrink" in lname:
                    self.has_gap_reduction = True

    def _scan_linked_list_operation(self, tree):
        """
        Distinguishes which Linked List operation is being performed,
        purely from structure - no naming/keyword matching.

        CRITICAL SCOPING: only scans inside top-level function bodies
        (the actual algorithm implementations), never top-level module
        script/setup code. Without this, boilerplate test-setup lines
        like `head = Node(10)` (needed to build a list before calling
        delete/reverse on it) would look identical to a real insert
        operation, since they also construct nodes - this would force
        every sample toward "insert" regardless of the actual function.
        """
        class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

        top_level_functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not top_level_functions:
            return

        search_nodes = [n for func in top_level_functions for n in ast.walk(func)]

        has_next_attr_access = any(
            isinstance(n, ast.Attribute) and n.attr == "next" for n in search_nodes
        )
        if not has_next_attr_access:
            return

        has_constructor_call = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in class_names
            for n in search_nodes
        )

        if has_constructor_call:
            self.has_ll_insert_pattern = True
            return

        has_skip_link = any(
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Attribute)
            and n.targets[0].attr == "next"
            and isinstance(n.value, ast.Attribute)
            and n.value.attr == "next"
            for n in search_nodes
        )

        if has_skip_link:
            self.has_ll_delete_pattern = True
            return

        for n in search_nodes:
            if isinstance(n, (ast.While, ast.For)):
                reassigned_names = set()
                for stmt in ast.walk(ast.Module(body=n.body, type_ignores=[])):
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        target = stmt.targets[0]
                        if isinstance(target, ast.Name):
                            reassigned_names.add(target.id)
                if len(reassigned_names) >= 3:
                    self.has_ll_reverse_pattern = True
                    return

    def _scan_union_find(self, tree):
        """
        Union-Find / Disjoint Set detection.
        SECONDARY heuristic: relies on function naming ('find'/'union')
        and a 'parent' array/dict, which is near-universal convention
        for this algorithm but not purely structural.
        """
        fn_names = {n.name.lower() for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        has_find = any("find" in name for name in fn_names)
        has_union = any("union" in name for name in fn_names)
        has_parent_var = any(
            isinstance(n, ast.Name) and "parent" in n.id.lower()
            for n in ast.walk(tree)
        )
        if (has_find and has_union) or (has_parent_var and (has_find or has_union)):
            self.has_union_find_pattern = True

    def _scan_topological_sort(self, tree):
        """
        Topological sort detection.
        SECONDARY heuristic: an 'indegree'-like variable combined with
        queue usage (collections.deque) is the standard Kahn's algorithm
        signature.
        """
        has_indegree_var = any(
            isinstance(n, ast.Name) and "indegree" in n.id.lower().replace("_", "")
            for n in ast.walk(tree)
        )
        has_deque_call = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "deque"
            for n in ast.walk(tree)
        )
        if has_indegree_var and has_deque_call:
            self.has_topological_sort_pattern = True

    # ================= Imports =================
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    # ================= Function defs / recursion =================
    def visit_FunctionDef(self, node):
        self.num_function_defs += 1
        if not self.current_function_stack:
            self.param_count = len(node.args.args)

        lname = node.name.lower()
        if any(k in lname for k in ("heapify", "sift", "percolate", "heapsort", "heap_sort")):
            self.has_heapify_pattern = True
        if "partition" in lname:
            self.has_pivot_partition = True
        if any(k in lname for k in ("radix", "count_sort", "counting_pass")):
            self.has_radix_exp_scaling = True

        # mutable-default-arg memoization pattern: def f(n, memo={}):
        defaults = node.args.defaults
        arg_names = [a.arg for a in node.args.args]
        default_arg_names = arg_names[len(arg_names) - len(defaults):] if defaults else []
        for name, default_val in zip(default_arg_names, defaults):
            if isinstance(default_val, (ast.Dict, ast.List)):
                self.dict_vars.add(name)
                self.memo_like_vars.add(name)  # default-arg cache is a strong memoization signal

        self.function_names.add(node.name)
        self.current_function_stack.append(node.name)
        self.generic_visit(node)
        self.current_function_stack.pop()

    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # recursion: call to the function we're currently inside
        if func_name and self.current_function_stack:
            current_fn = self.current_function_stack[-1]
            if func_name == current_fn:
                self.recursive_calls[current_fn] = self.recursive_calls.get(current_fn, 0) + 1

        if func_name in ("heappush", "heappop", "heapify", "nlargest", "nsmallest"):
            self.uses_priority_queue = True
        if func_name in ("heapify", "sift", "percolate"):
            self.has_heapify_pattern = True
        if func_name == "deque":
            self.uses_queue = True
        if func_name in ("sorted", "sort"):
            self.uses_sorted_or_sort_call = True

        # Sliced recursive call check (e.g. merge sort: func(arr[:mid]), func(arr[mid:]))
        if any(isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice) for arg in node.args):
            self.has_split_slicing = True

        self.generic_visit(node)

    # ================= Loops =================
    def _enter_loop(self):
        self._current_loop_depth += 1
        self.num_loops += 1
        self.max_loop_depth = max(self.max_loop_depth, self._current_loop_depth)
        self._inside_loop_depth_for_if += 1

    def _exit_loop(self):
        self._current_loop_depth -= 1
        self._inside_loop_depth_for_if -= 1

    def visit_For(self, node):
        self._enter_loop()
        self._check_swap_in_body(node.body)
        self._check_backtracking_undo(node.body)
        self._check_sliding_window(node)
        self._check_monotonic_stack(node.body)
        self._check_same_direction_pointer(node)
        self._check_key_and_shifting_in_loop(node.body)
        self._check_auxiliary_merging_in_body(node.body)
        self._check_pivot_in_body(node.body)
        self.generic_visit(node)
        self._exit_loop()

    def visit_While(self, node):
        self._enter_loop()
        self._check_swap_in_body(node.body)
        self._check_two_pointer(node)
        self._check_monotonic_stack(node.body)
        self._check_parallel_two_pointer(node)
        self._check_key_and_shifting_in_loop(node.body)
        self._check_auxiliary_merging_in_body(node.body)
        self._check_gap_in_while(node)
        self.generic_visit(node)
        self._exit_loop()

    def _check_gap_in_while(self, node):
        """Checks if while loop test is driven by a gap variable e.g. while gap > 0:"""
        if isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and "gap" in left.id.lower():
                self.has_gap_reduction = True
        elif isinstance(node.test, ast.BoolOp):
            for v in node.test.values:
                if isinstance(v, ast.Compare) and isinstance(v.left, ast.Name) and "gap" in v.left.id.lower():
                    self.has_gap_reduction = True

    def _check_key_and_shifting_in_loop(self, body):
        """Detects insertion sort / shell sort shifting and key element assignments."""
        for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(stmt, ast.Assign):
                # key = arr[i]
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
                    t_name = stmt.targets[0].id.lower()
                    if any(k in t_name for k in ("key", "tmp", "temp", "val", "item", "curr")):
                        self.has_key_assignment = True
                    elif isinstance(stmt.value.slice, (ast.Name, ast.BinOp)):
                        self.has_key_assignment = True

                # arr[j + 1] = arr[j] or arr[j] = arr[j - gap]
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript) and isinstance(stmt.value, ast.Subscript):
                    t_sub = stmt.targets[0]
                    v_sub = stmt.value
                    if isinstance(t_sub.value, ast.Name) and isinstance(v_sub.value, ast.Name):
                        if t_sub.value.id == v_sub.value.id:
                            self.has_element_shifting = True

    def _check_auxiliary_merging_in_body(self, body):
        """Detects appending or extending to an auxiliary list from split segments."""
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr in ("append", "extend"):
                    if any(isinstance(arg, ast.Subscript) for arg in n.args):
                        self.has_auxiliary_merging = True

    def _check_pivot_in_body(self, body):
        """Detects pivot comparison (e.g. if arr[j] <= pivot:)."""
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.Compare):
                for node in (n.left, *n.comparators):
                    if isinstance(node, ast.Name) and "pivot" in node.id.lower():
                        self.has_pivot_partition = True

    def _check_swap_in_body(self, body):
        """Tuple-swap pattern: a[i], a[j] = a[j], a[i] (searched recursively,
        since it may be nested inside an if-block)."""
        for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Tuple):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Tuple):
                    self.has_swap_in_loop = True

    _TWO_POINTER_NAMES = {
        "lo", "hi", "low", "high", "left", "right", "l", "r", "start", "end", "i", "j",
        # additional aliases seen in real naming conventions / dataset generator pools
        "left_ptr", "right_ptr", "s", "e", "begin", "finish",
        "start_idx", "end_idx", "from_idx", "to_idx",
    }

    def _check_two_pointer(self, node):
        """Heuristic: while loop whose condition compares two variables
        both drawn from conventional two-pointer names (lo/hi, left/right,
        start/end, etc.) — narrower than 'any Name on the left' to avoid
        false positives on unrelated while-loops (e.g. `while exp > 0`)."""
        if isinstance(node, ast.While) and isinstance(node.test, ast.Compare):
            left = node.test.left
            comparators = node.test.comparators
            names_in_test = []
            if isinstance(left, ast.Name):
                names_in_test.append(left.id.lower())
            for c in comparators:
                if isinstance(c, ast.Name):
                    names_in_test.append(c.id.lower())
            if len(names_in_test) >= 2 and all(n in self._TWO_POINTER_NAMES for n in names_in_test):
                self.has_two_pointer_pattern = True

    def _check_backtracking_undo(self, body):
        """Detects the classic backtracking shape, searched recursively since
        it commonly appears nested inside an `if` block within the loop:
            path.append(x)
            backtrack(...)   # recursive call
            path.pop()
        """
        self._scan_backtracking_block(body)

    def _scan_backtracking_block(self, body):
        add_methods = ("append", "add")
        remove_methods = ("pop", "remove")
        add_idx, add_var, recurse_idx = None, None, None

        for idx, stmt in enumerate(body):
            call = stmt.value if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) else None

            if call is not None:
                if isinstance(call.func, ast.Attribute) and call.func.attr in add_methods:
                    if isinstance(call.func.value, ast.Name):
                        add_idx, add_var = idx, call.func.value.id

                elif isinstance(call.func, ast.Name) and self.current_function_stack:
                    if call.func.id == self.current_function_stack[-1]:
                        recurse_idx = idx

                elif isinstance(call.func, ast.Attribute) and call.func.attr in remove_methods:
                    if (
                        isinstance(call.func.value, ast.Name)
                        and call.func.value.id == add_var
                        and add_idx is not None
                        and recurse_idx is not None
                        and add_idx < recurse_idx < idx
                    ):
                        self.has_backtracking_undo_pattern = True

            # Recurse into nested blocks (if/for/while bodies) since the
            # append -> recurse -> pop sequence is very commonly wrapped
            # in a condition (e.g. "if is_valid(...): ...").
            if isinstance(stmt, ast.If):
                self._scan_backtracking_block(stmt.body)
                self._scan_backtracking_block(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.While)):
                self._scan_backtracking_block(stmt.body)

    def _check_sliding_window(self, outer_node):
        """Heuristic: a for-loop containing a nested while-loop whose
        condition is an arithmetic comparison (e.g. `end - start + 1 > k`) —
        the classic window-shrink condition. Deliberately narrower than
        'any nested while' so it doesn't collide with monotonic-stack loops,
        which use a BoolOp condition like `stack and nums[stack[-1]] < x`."""
        for n in ast.walk(outer_node):
            if isinstance(n, ast.While) and n is not outer_node:
                if isinstance(n.test, ast.Compare):
                    if isinstance(n.test.left, ast.BinOp) or any(
                        isinstance(c, ast.BinOp) for c in n.test.comparators
                    ):
                        self.has_sliding_window_pattern = True

    def _check_monotonic_stack(self, body):
        """Detects: while stack and <comparison>: stack.pop()  ... stack.append(...)
        i.e. popping from a stack conditionally before pushing, within the same loop body."""
        has_conditional_pop = False
        has_append = False
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.While) and isinstance(n.test, ast.BoolOp):
                for sub in ast.walk(n):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "pop"
                    ):
                        has_conditional_pop = True
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
            ):
                has_append = True
        if has_conditional_pop and has_append:
            self.has_monotonic_stack_pattern = True

    def _check_same_direction_pointer(self, for_node):
        """Detects the classic 'same-direction' two-pointer shape used for
        in-place array compaction (e.g. remove duplicates):
            for j in range(1, len(arr)):
                if <condition>:
                    i += 1           # secondary index advances conditionally
                    arr[i] = arr[j]  # used to overwrite in place
        Distinguished from the opposite-ends pattern (which is while-loop
        based) because here a FOR loop drives one index, and a SEPARATE
        index variable only advances conditionally inside an if-block and
        is used as a write-target subscript.
        """
        loop_var = for_node.target.id if isinstance(for_node.target, ast.Name) else None
        if loop_var is None:
            return

        for n in ast.walk(ast.Module(body=for_node.body, type_ignores=[])):
            if not isinstance(n, ast.If):
                continue
            incremented_vars = set()
            written_subscript_vars = set()
            for stmt in ast.walk(ast.Module(body=n.body, type_ignores=[])):
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                    if isinstance(stmt.op, ast.Add):
                        incremented_vars.add(stmt.target.id)
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Subscript):
                            idx_node = target.slice
                            # Python 3.9+: slice is the index expression directly
                            if isinstance(idx_node, ast.Name):
                                written_subscript_vars.add(idx_node.id)
            secondary = (incremented_vars & written_subscript_vars) - {loop_var}
            if secondary:
                self.has_same_direction_pointer_pattern = True

    def _check_parallel_two_pointer(self, while_node):
        """Detects the 'parallel pointers over two arrays' shape:
            while i < len(arr1) and j < len(arr2):
                ...
        i.e. a while-loop whose test is an `and` of two separate Compare
        clauses, each bounding a different index variable — the classic
        merge-two-sorted-arrays / merge-two-sorted-lists signature.
        """
        test = while_node.test
        if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.And):
            return
        if len(test.values) < 2:
            return
        index_names = set()
        for clause in test.values:
            if isinstance(clause, ast.Compare) and isinstance(clause.left, ast.Name):
                index_names.add(clause.left.id)
        if len(index_names) >= 2:
            self.has_parallel_two_pointer_pattern = True
    def visit_If(self, node):
        if self._inside_loop_depth_for_if > 0:
            self._current_if_nesting += 1
            self.max_if_nesting_in_loop = max(self.max_if_nesting_in_loop, self._current_if_nesting)
            self.generic_visit(node)
            self._current_if_nesting -= 1
        else:
            self.generic_visit(node)

    def visit_Return(self, node):
        if self._current_loop_depth > 0:
            self.has_early_return_in_loop = True
        self.generic_visit(node)

    # ================= Assignments =================
    def visit_Assign(self, node):
        self._check_fast_slow_pointer(node)
        self._check_matrix_transpose(node)
        self._check_prefix_pattern(node)
        self._check_data_structure_vars(node)
        self._check_lomuto_hoare(node)
        self._check_frequency_array(node)
        self._check_gap_assign(node)
        self.generic_visit(node)

    def _check_gap_assign(self, node):
        """gap = n // 2 or gap = int(gap / 1.3)"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id.lower()
            if "gap" in name or "shrink" in name:
                self.has_gap_reduction = True

    def _check_frequency_array(self, node):
        """Pattern: count = [0] * span, count = [0] * 10, holes = [[] for _ in range(...)]"""
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            target_name = node.targets[0].id.lower() if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else ""
            mult_val = node.value.right if isinstance(node.value.left, ast.List) else (node.value.left if isinstance(node.value.right, ast.List) else None)
            is_zero_list = False
            if isinstance(node.value.left, ast.List) and len(node.value.left.elts) == 1 and isinstance(node.value.left.elts[0], ast.Constant) and node.value.left.elts[0].value == 0:
                is_zero_list = True
            elif isinstance(node.value.right, ast.List) and len(node.value.right.elts) == 1 and isinstance(node.value.right.elts[0], ast.Constant) and node.value.right.elts[0].value == 0:
                is_zero_list = True

            if is_zero_list:
                if any(k in target_name for k in ("count", "freq", "bucket", "hole", "radix", "span")):
                    self.has_frequency_array = True
                elif isinstance(mult_val, ast.Constant) and mult_val.value in (10, 256, 128, 16):
                    self.has_frequency_array = True
                elif isinstance(mult_val, ast.Name) and any(k in mult_val.id.lower() for k in ("span", "range", "max", "k")):
                    self.has_frequency_array = True
                elif isinstance(mult_val, ast.BinOp):
                    self.has_frequency_array = True
        elif isinstance(node.value, ast.ListComp):
            if isinstance(node.value.elt, ast.List) and len(node.value.elt.elts) == 0:
                target_name = node.targets[0].id.lower() if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else ""
                if any(k in target_name for k in ("bucket", "hole", "bin", "count")):
                    self.has_frequency_array = True

    def _check_lomuto_hoare(self, node):
        """i = start - 1 or arr[i + 1], arr[end] = arr[end], arr[i + 1]"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Sub):
                if isinstance(node.value.right, ast.Constant) and node.value.right.value == 1:
                    if isinstance(node.value.left, ast.Name):
                        name = node.value.left.id.lower()
                        if name in ("start", "low", "l", "begin", "lo", "left"):
                            self.has_lomuto_hoare_indexing = True
        elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            elts = node.targets[0].elts
            if len(elts) == 2:
                # Check for arr[i+1], arr[end] (one has BinOp + 1, other is variable like end/high/start/pivot)
                s0 = elts[0].slice if isinstance(elts[0], ast.Subscript) else None
                s1 = elts[1].slice if isinstance(elts[1], ast.Subscript) else None
                if s0 and s1:
                    has_add1 = (isinstance(s0, ast.BinOp) and isinstance(s0.op, ast.Add) and isinstance(s0.right, ast.Constant) and s0.right.value == 1) or \
                               (isinstance(s1, ast.BinOp) and isinstance(s1.op, ast.Add) and isinstance(s1.right, ast.Constant) and s1.right.value == 1)
                    has_boundary = (isinstance(s0, ast.Name) and s0.id.lower() in ("end", "high", "hi", "pivot", "start", "low", "lo", "right", "left")) or \
                                   (isinstance(s1, ast.Name) and s1.id.lower() in ("end", "high", "hi", "pivot", "start", "low", "lo", "right", "left"))
                    if has_add1 and has_boundary:
                        self.has_lomuto_hoare_indexing = True

    def _check_fast_slow_pointer(self, node):
        """Pattern: fast = fast.next.next (double attribute access),
        the hallmark of Floyd's cycle detection ('fast' pointer)."""
        if isinstance(node.value, ast.Attribute) and node.value.attr == "next":
            inner = node.value.value
            if isinstance(inner, ast.Attribute) and inner.attr == "next":
                self.has_fast_slow_pointer_pattern = True

    def _check_matrix_transpose(self, node):
        """Pattern: matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]"""
        if isinstance(node.value, ast.Tuple) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and len(target.elts) == 2:
                t0, t1 = target.elts
                if (
                    isinstance(t0, ast.Subscript) and isinstance(t1, ast.Subscript)
                    and isinstance(t0.value, ast.Subscript) and isinstance(t1.value, ast.Subscript)
                ):
                    self.has_matrix_transpose_pattern = True

    def _check_prefix_pattern(self, node):
        """Pattern: arr[i] = arr[i-1] + <something>"""
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(node.value, ast.BinOp):
                if isinstance(node.value.op, ast.Add):
                    base = target.value
                    left = node.value.left
                    if (
                        isinstance(base, ast.Name)
                        and isinstance(left, ast.Subscript)
                        and isinstance(left.value, ast.Name)
                        and left.value.id == base.id
                    ):
                        self.has_prefix_pattern = True

    def _check_data_structure_vars(self, node):
        for target in node.targets:
            var_name = target.id if isinstance(target, ast.Name) else None

            if var_name:
                lname = var_name.lower()
                if any(k in lname for k in ("visited", "seen", "marked")):
                    self.has_visited_like_var = True
                if any(k in lname for k in ("adj", "graph", "edges", "neighbors", "neighbours")):
                    self.uses_adjacency_structure = True
                if any(k in lname for k in ("memo", "cache", "dp")):
                    self.memo_like_vars.add(var_name)
                if "pivot" in lname:
                    self.has_pivot_partition = True

            if isinstance(node.value, ast.Dict) and var_name:
                self.dict_vars.add(var_name)
            elif isinstance(node.value, ast.Set) and var_name:
                self.set_vars.add(var_name)
            elif isinstance(node.value, (ast.List, ast.ListComp)) and var_name:
                self.list_vars.add(var_name)
            elif isinstance(node.value, ast.Call) and var_name:
                callee = node.value.func.id if isinstance(node.value.func, ast.Name) else None
                if callee in ("dict", "defaultdict"):
                    self.dict_vars.add(var_name)
                elif callee == "set":
                    self.set_vars.add(var_name)
                elif callee == "list":
                    self.list_vars.add(var_name)

    # ================= Subscript (array indexing depth & slicing) =================
    def visit_Subscript(self, node):
        depth = 0
        cur = node
        while isinstance(cur, ast.Subscript):
            depth += 1
            cur = cur.value
        self.max_subscript_depth = max(self.max_subscript_depth, depth)
        if depth >= 2:
            self.has_2d_list_indexing = True

        # Slicing detection: [:mid], [mid:], [mid+1:], [:len(arr)//2]
        if isinstance(node.slice, ast.Slice):
            sl = node.slice
            if sl.upper is not None and sl.lower is None:
                if isinstance(sl.upper, ast.Name) and any(k in sl.upper.id.lower() for k in ("mid", "half", "m", "center", "k")):
                    self.has_split_slicing = True
                elif isinstance(sl.upper, ast.BinOp) and isinstance(sl.upper.op, (ast.FloorDiv, ast.Div, ast.Add, ast.Sub)):
                    self.has_split_slicing = True
            elif sl.lower is not None and sl.upper is None:
                if isinstance(sl.lower, ast.Name) and any(k in sl.lower.id.lower() for k in ("mid", "half", "m", "center", "k")):
                    self.has_split_slicing = True
                elif isinstance(sl.lower, ast.BinOp) and isinstance(sl.lower.op, (ast.FloorDiv, ast.Div, ast.Add, ast.Sub)):
                    self.has_split_slicing = True

        self.generic_visit(node)

    # ================= Bitwise / modular / child math =================
    def visit_BinOp(self, node):
        if isinstance(node.op, _BITWISE_OPS):
            self.uses_bit_manipulation = True
        if isinstance(node.op, ast.Mod):
            self.uses_modular_math = True
            if isinstance(node.right, ast.Constant) and node.right.value == 10:
                self.has_radix_exp_scaling = True

        self._check_midpoint_calc(node)
        self._check_heap_child_math(node)
        self.generic_visit(node)

    def _check_heap_child_math(self, node):
        """Detects child math: 2*i + 1, 2*i + 2, 2*idx + 1, etc."""
        if isinstance(node.op, ast.Add):
            left, right = node.left, node.right
            if isinstance(right, ast.Constant) and right.value in (1, 2):
                if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mult):
                    if (isinstance(left.left, ast.Constant) and left.left.value == 2) or \
                       (isinstance(left.right, ast.Constant) and left.right.value == 2):
                        self.has_heap_child_math = True
            elif isinstance(left, ast.Constant) and left.value in (1, 2):
                if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult):
                    if (isinstance(right.left, ast.Constant) and right.left.value == 2) or \
                       (isinstance(right.right, ast.Constant) and right.right.value == 2):
                        self.has_heap_child_math = True

    def _check_midpoint_calc(self, node):
        if not isinstance(node.op, (ast.FloorDiv, ast.Div)):
            return
        if not (isinstance(node.right, ast.Constant) and node.right.value == 2):
            return
        left = node.left
        if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Add):
            self.has_midpoint_calculation = True

    def visit_AugAssign(self, node):
        if isinstance(node.op, _BITWISE_OPS):
            self.uses_bit_manipulation = True
        if isinstance(node.op, ast.Mod):
            self.uses_modular_math = True
        if isinstance(node.op, (ast.FloorDiv, ast.Div)):
            if isinstance(node.target, ast.Name):
                name = node.target.id.lower()
                if "gap" in name or "shrink" in name or "step" in name or name == "h":
                    self.has_gap_reduction = True
                elif self._current_loop_depth > 0:
                    self.has_gap_reduction = True
        if isinstance(node.op, ast.Mult):
            if isinstance(node.target, ast.Name):
                name = node.target.id.lower()
                if "exp" in name or "radix" in name or "digit" in name:
                    self.has_radix_exp_scaling = True
                elif isinstance(node.value, ast.Constant) and node.value.value == 10:
                    self.has_radix_exp_scaling = True
        self.generic_visit(node)

    # ================= Linked list / tree =================
    def visit_ClassDef(self, node):
        assigned_self_attrs = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        assigned_self_attrs.add(target.attr)
        if "next" in assigned_self_attrs:
            self.has_linked_list_pattern = True
        if any(a in assigned_self_attrs for a in ("left", "right", "children")):
            self.has_tree_pattern = True
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == "next":
            self.has_linked_list_pattern = True
        elif node.attr in ("left", "right"):
            self.has_tree_pattern = True
        self.generic_visit(node)


def extract_features(code: str) -> Dict[str, float]:
    """
    Parse Python source `code` and return a dict of numeric features.
    Raises SyntaxError if the code cannot be parsed.
    """
    tree = ast.parse(code)
    visitor = _StructuralVisitor()
    visitor.visit(tree)

    has_recursion = 1 if any(c > 0 for c in visitor.recursive_calls.values()) else 0
    recursion_branch_count = max(visitor.recursive_calls.values(), default=0)

    has_memoization = 0
    if has_recursion and visitor.memo_like_vars:
        has_memoization = 1

    uses_stack = 1 if visitor.uses_stack_lib else 0
    has_dual_recursion = 1 if (recursion_branch_count >= 2 or visitor.has_dual_recursion) else 0

    features = {
        "max_loop_depth": float(visitor.max_loop_depth),
        "num_loops": float(visitor.num_loops),
        "has_recursion": float(has_recursion),
        "recursion_branch_count": float(recursion_branch_count),
        "has_memoization": float(has_memoization),
        "uses_priority_queue": float(visitor.uses_priority_queue),
        "uses_queue": float(visitor.uses_queue),
        "uses_stack": float(uses_stack),
        "uses_set": float(len(visitor.set_vars) > 0),
        "uses_dict": float(len(visitor.dict_vars) > 0),
        "has_visited_like_var": float(visitor.has_visited_like_var),
        "has_2d_list_indexing": float(visitor.has_2d_list_indexing),
        "array_dimensionality": float(visitor.max_subscript_depth),
        "has_swap_in_loop": float(visitor.has_swap_in_loop),
        "uses_sorted_or_sort_call": float(visitor.uses_sorted_or_sort_call),
        "has_two_pointer_pattern": float(visitor.has_two_pointer_pattern),
        "num_function_defs": float(visitor.num_function_defs),
        "max_if_nesting_in_loop": float(visitor.max_if_nesting_in_loop),
        "has_early_return_in_loop": float(visitor.has_early_return_in_loop),
        "param_count": float(visitor.param_count),
        "uses_adjacency_structure": float(visitor.uses_adjacency_structure),
        "uses_bit_manipulation": float(visitor.uses_bit_manipulation),
        "has_linked_list_pattern": float(visitor.has_linked_list_pattern),
        "has_ll_insert_pattern": float(visitor.has_ll_insert_pattern),
        "has_ll_delete_pattern": float(visitor.has_ll_delete_pattern),
        "has_ll_reverse_pattern": float(visitor.has_ll_reverse_pattern),
        "has_tree_pattern": float(visitor.has_tree_pattern),
        "has_prefix_pattern": float(visitor.has_prefix_pattern),
        "has_backtracking_undo_pattern": float(visitor.has_backtracking_undo_pattern),
        "has_union_find_pattern": float(visitor.has_union_find_pattern),
        "has_fast_slow_pointer_pattern": float(visitor.has_fast_slow_pointer_pattern),
        "has_sliding_window_pattern": float(visitor.has_sliding_window_pattern),
        "has_monotonic_stack_pattern": float(visitor.has_monotonic_stack_pattern),
        "has_topological_sort_pattern": float(visitor.has_topological_sort_pattern),
        "has_matrix_transpose_pattern": float(visitor.has_matrix_transpose_pattern),
        "uses_modular_math": float(visitor.uses_modular_math),
        "has_same_direction_pointer_pattern": float(visitor.has_same_direction_pointer_pattern),
        "has_parallel_two_pointer_pattern": float(visitor.has_parallel_two_pointer_pattern),
        "has_midpoint_calculation": float(visitor.has_midpoint_calculation),
        # Algorithm-specific structural AST signals for sorting
        "has_dual_recursion": float(has_dual_recursion),
        "has_pivot_partition": float(visitor.has_pivot_partition),
        "has_lomuto_hoare_indexing": float(visitor.has_lomuto_hoare_indexing),
        "has_split_slicing": float(visitor.has_split_slicing),
        "has_auxiliary_merging": float(visitor.has_auxiliary_merging),
        "has_key_assignment": float(visitor.has_key_assignment),
        "has_element_shifting": float(visitor.has_element_shifting),
        "has_gap_reduction": float(visitor.has_gap_reduction),
        "has_heapify_pattern": float(1 if (visitor.has_heapify_pattern or visitor.has_heap_child_math) else 0),
        "has_heap_child_math": float(visitor.has_heap_child_math),
        "has_frequency_array": float(visitor.has_frequency_array),
        "has_radix_exp_scaling": float(visitor.has_radix_exp_scaling),
    }
    return features


if __name__ == "__main__":
    sample = """
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""
    import json
    print(json.dumps(extract_features(sample), indent=2))