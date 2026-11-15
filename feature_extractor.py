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
import re
from typing import Dict, Any, Optional, List


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
    # Search-specific features to distinguish Binary Search from Merge Sort
    "has_target_comparison",
    "has_directional_pointer_adjustment",
    "has_single_path_search",
    # BFS discriminative features
    "has_queue_while_loop",
    "has_queue_pop",
    "has_visited_check_before_enqueue",
    "has_adjacency_iteration",
    "has_dp_table_indexing",
    "has_memo_dict_access",
    # Bucket Sort structural features
    "has_bucket_comprehension",
    "has_bucket_index_calculation",
    "has_bucket_sort_loop",
    "has_bucket_concatenation",
    # Prim / Dijkstra / MST discriminative features
    "has_priority_queue_traversal",
    "has_weighted_adjacency",
    "has_mst_weight_accumulation",
    "has_mst_edge_building",
    "has_lazy_deletion_on_pop",
    "has_dist_relaxation",
    # Topological Sort discriminative features
    "has_topo_dfs_postorder",
    "has_three_color_cycle_detection",
    "has_reversed_stack_return",
    # Greedy / Interval Scheduling discriminative features
    "has_interval_scheduling_pattern",
    "has_interval_sort_key",
    "has_interval_compatibility_check",
    # Word Break vs Counting Sort discriminative features
    "has_element_indexed_count",
    "has_count_sort_rebuild",
    "has_string_slicing_subscript",
    "has_earlier_dp_dependency",
    "has_word_break_pattern",
    # Graph family discriminative features
    "has_reversed_neighbor_traversal",
    "has_stack_extend_neighbors",
    "has_deque_appendleft",
    "has_bellman_ford_edges_loop",
    "has_negative_cycle_check",
    "has_apsp_nested_sources",
    "has_dag_relaxation_loop",
    "has_cycle_path_set",
    "has_zero_one_weights",
    "has_topological_sort_kahn",
    "has_topological_sort_dfs",
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
        
        # Search-specific features to distinguish Binary Search from Merge Sort
        self.has_target_comparison = False
        self.has_directional_pointer_adjustment = False
        self.has_single_path_search = False

        # BFS discriminative features
        self.has_queue_while_loop = False
        self.has_queue_pop = False
        self.has_visited_check_before_enqueue = False
        self.has_adjacency_iteration = False
        self.has_dp_table_indexing = False
        self.has_memo_dict_access = False

        # Prim / Dijkstra / MST structural signals
        self.has_priority_queue_traversal = False
        self.has_weighted_adjacency = False
        self.has_mst_weight_accumulation = False
        self.has_mst_edge_building = False
        self.has_lazy_deletion_on_pop = False
        self.has_dist_relaxation = False

        # Bucket Sort structural features
        self.has_bucket_comprehension = False
        self.has_bucket_index_calculation = False
        self.has_bucket_sort_loop = False
        self.has_bucket_concatenation = False

        # Topological Sort structural features
        self.has_topo_dfs_postorder = False
        self.has_three_color_cycle_detection = False
        self.has_reversed_stack_return = False

        # Greedy / Interval Scheduling structural features
        self.has_interval_scheduling_pattern = False
        self.has_interval_sort_key = False
        self.has_interval_compatibility_check = False

        # Word Break vs Counting Sort structural features
        self.has_element_indexed_count = False
        self.has_count_sort_rebuild = False
        self.has_string_slicing_subscript = False
        self.has_earlier_dp_dependency = False
        self.has_word_break_pattern = False

        # Graph family structural features
        self.has_reversed_neighbor_traversal = False
        self.has_stack_extend_neighbors = False
        self.has_deque_appendleft = False
        self.has_bellman_ford_edges_loop = False
        self.has_negative_cycle_check = False
        self.has_apsp_nested_sources = False
        self.has_dag_relaxation_loop = False
        self.has_cycle_path_set = False
        self.has_zero_one_weights = False
        self.has_topological_sort_kahn = False
        self.has_topological_sort_dfs = False

        self._union_find_scanned = False

    # ================= Module-level pre-scan =================
    def visit_Module(self, node):
        self._scan_bfs_structures(node)  # sets adjacency and queue structures first
        self._scan_graph_family(node)
        self._scan_union_find(node)
        self._scan_topological_sort(node)
        self._scan_interval_scheduling_structures(node)
        self._scan_word_break_structures(node)
        self._scan_counting_sort_structures(node)
        self._scan_linked_list_operation(node)
        self._scan_sorting_structures(node)
        self._scan_binary_search_structure(node)  # must run before loop visitors
        self._scan_bucket_sort_structures(node)
        self._scan_prim_mst_structures(node)
        self.generic_visit(node)

    def _scan_bfs_structures(self, tree):
        """
        Extracts unambiguous BFS queue and graph frontier signals:
        - while loop over a queue/deque
        - pop(0) or popleft()
        - visited set membership check before enqueueing
        - adjacency iteration (graph[node] or adj[u])
        - absence/presence of 2D DP table indexing (dp[i][j]) and memo dict access
        """
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                for a in n.args.args:
                    aname = a.arg.lower()
                    if any(k in aname for k in ("graph", "adj", "edges", "neighbors", "neighbours")):
                        self.uses_adjacency_structure = True

            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "popleft":
                    self.has_queue_pop = True
                    self.uses_queue = True
                elif n.func.attr == "pop" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 0:
                    self.has_queue_pop = True
                    self.uses_queue = True

            if isinstance(n, ast.While):
                has_pop = any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and (
                        c.func.attr == "popleft" or
                        (c.func.attr == "pop" and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == 0)
                    )
                    for c in ast.walk(n)
                )
                if has_pop:
                    self.has_queue_while_loop = True
                    self.uses_queue = True
                elif isinstance(n.test, ast.Name) and any(k in n.test.id.lower() for k in ("queue", "q", "frontier")):
                    # Do not classify as BFS queue if this while loop actually operates on a min-heap / heappop
                    has_heappop = any(
                        isinstance(c, ast.Call) and (
                            (isinstance(c.func, ast.Name) and c.func.id == "heappop") or
                            (isinstance(c.func, ast.Attribute) and c.func.attr == "heappop")
                        )
                        for c in ast.walk(n)
                    )
                    if not has_heappop:
                        self.has_queue_while_loop = True
                        self.uses_queue = True

            if isinstance(n, ast.For):
                if isinstance(n.iter, ast.Subscript):
                    if isinstance(n.iter.value, ast.Name):
                        vname = n.iter.value.id.lower()
                        if any(k in vname for k in ("graph", "adj", "edges", "network", "node")):
                            self.has_adjacency_iteration = True
                            self.uses_adjacency_structure = True
                        elif self.uses_adjacency_structure:
                            self.has_adjacency_iteration = True
                elif isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute):
                    if n.iter.func.attr in ("get", "neighbors", "items", "values"):
                        if isinstance(n.iter.func.value, ast.Name) and any(k in n.iter.func.value.id.lower() for k in ("graph", "adj", "edges", "network")):
                            self.has_adjacency_iteration = True
                            self.uses_adjacency_structure = True

            if isinstance(n, ast.If):
                is_visited_check = False
                if isinstance(n.test, ast.Compare):
                    for op in n.test.ops:
                        if isinstance(op, ast.NotIn):
                            is_visited_check = True
                elif isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not):
                    is_visited_check = True

                if is_visited_check:
                    has_enqueue = any(
                        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("append", "appendleft", "extend")
                        for c in ast.walk(n)
                    )
                    if has_enqueue:
                        self.has_visited_check_before_enqueue = True
                        self.has_visited_like_var = True

            if isinstance(n, ast.Subscript):
                if isinstance(n.value, ast.Subscript):
                    base = n.value.value
                    if isinstance(base, ast.Name) and any(k in base.id.lower() for k in ("dp", "table", "opt", "memo")):
                        self.has_dp_table_indexing = True
                elif isinstance(n.value, ast.Name) and any(k in n.value.id.lower() for k in ("dp", "table", "opt")):
                    if isinstance(n.slice, ast.BinOp) and isinstance(n.slice.op, (ast.Sub, ast.Add)):
                        self.has_dp_table_indexing = True

            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                if any(k in n.value.id.lower() for k in ("memo", "cache")):
                    self.has_memo_dict_access = True

    def _scan_graph_family(self, tree):
        """
        Extracts structural features across the graph algorithm family:
        - BFS vs DFS (iterative reversed, stack.extend, recursive explore)
        - Dijkstra vs Prim (distance relaxation vs cut-set weight accumulation)
        - Bellman-Ford (V-1 passes over edges, negative cycle check)
        - 0-1 BFS (deque with appendleft on weight 0, append on 1)
        - APSP (repeated Bellman-Ford / SSSP per source)
        - Topological Sort (Kahn in-degree vs DFS post-order with path set cycle detection)
        - DAG Shortest Path (topological order + edge relaxation)
        """
        has_reversed = False
        has_extend = False
        has_appendleft = False
        has_weight_zero = False
        has_v_minus_1 = False
        has_edge_iteration = False
        has_neg_cycle_check = False
        has_apsp_sources = False
        has_dag_relax = False
        has_path_add = False
        has_path_remove = False
        has_indegree = False

        for n in ast.walk(tree):
            # 1. Check adjacency subscript access: graph[node], adj[u], etc.
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                vname = n.value.id.lower()
                if any(k in vname for k in ("graph", "adj", "edges", "matrix", "network")):
                    self.uses_adjacency_structure = True

            # 2. Check loops over neighbors
            if isinstance(n, ast.For):
                # for neighbor in reversed(graph[node]):
                if isinstance(n.iter, ast.Call):
                    callee = n.iter.func.id if isinstance(n.iter.func, ast.Name) else ""
                    if callee == "reversed":
                        has_reversed = True
                        self.uses_adjacency_structure = True
                        self.has_adjacency_iteration = True
                elif isinstance(n.iter, ast.Subscript):
                    if isinstance(n.iter.value, ast.Name):
                        vname = n.iter.value.id.lower()
                        if any(k in vname for k in ("graph", "adj", "edges", "network", "node")):
                            self.uses_adjacency_structure = True
                            self.has_adjacency_iteration = True
                
                # Check V-1 loop: for _ in range(vertices - 1)
                if isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range":
                    if n.iter.args and isinstance(n.iter.args[0], ast.BinOp) and isinstance(n.iter.args[0].op, ast.Sub):
                        has_v_minus_1 = True
                
                # Check edge tuple iteration: for u, v, w in edges: or for u, v in edges:
                if isinstance(n.target, ast.Tuple) and len(n.target.elts) in (2, 3):
                    iter_name = getattr(n.iter, "id", "").lower()
                    if any(k in iter_name for k in ("edges", "edge", "adj", "graph", "pairs", "prerequisites")):
                        has_edge_iteration = True
                        self.uses_adjacency_structure = True

                # Check outer source iteration: for src in range(vertices) calling shortest path function
                if isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range":
                    for inner in ast.walk(n):
                        if isinstance(inner, ast.Call):
                            cname = inner.func.id if isinstance(inner.func, ast.Name) else ""
                            if any(k in cname.lower() for k in ("bellman", "dijkstra", "sssp", "sp")):
                                has_apsp_sources = True

                # Check DAG relaxation along topological order: for u in topo_order:
                if isinstance(n.iter, ast.Name):
                    iname = n.iter.id.lower()
                    if any(k in iname for k in ("topo", "linear", "order", "stack")):
                        for inner in ast.walk(n):
                            if isinstance(inner, ast.Compare):
                                cstr = ast.unparse(inner).lower() if hasattr(ast, "unparse") else ""
                                if "+" in cstr and "<" in cstr and "dist" in cstr:
                                    has_dag_relax = True

            # 3. Check calls: appendleft, extend, path.add/remove
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                attr = n.func.attr
                vname = getattr(n.func.value, "id", "").lower()
                if attr == "appendleft":
                    has_appendleft = True
                    self.uses_queue = True
                elif attr == "extend" and any(k in vname for k in ("stack", "stk", "order")):
                    has_extend = True
                    self.uses_stack = True
                    self.uses_adjacency_structure = True
                elif any(k in vname for k in ("path", "visiting", "cycle")):
                    if attr == "add":
                        has_path_add = True
                    elif attr == "remove":
                        has_path_remove = True

            # 4. Check comparisons for weight == 0
            if isinstance(n, ast.Compare):
                if isinstance(n.left, ast.Name) and n.left.id.lower() in ("weight", "w", "wt", "cost"):
                    for op, comp in zip(n.ops, n.comparators):
                        if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and comp.value == 0:
                            has_weight_zero = True

            # 5. Check in-degree tracking
            if isinstance(n, ast.Name) and "indegree" in n.id.lower().replace("_", ""):
                has_indegree = True

            # 6. Check negative cycle detection
            if isinstance(n, ast.If):
                cstr = ast.unparse(n.test).lower() if hasattr(ast, "unparse") else ""
                if "+" in cstr and "<" in cstr and ("dist" in cstr or "d[" in cstr):
                    for stmt in n.body:
                        if isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value in (None, False))):
                            has_neg_cycle_check = True

        self.has_reversed_neighbor_traversal = has_reversed
        self.has_stack_extend_neighbors = has_extend
        self.has_deque_appendleft = has_appendleft
        self.has_zero_one_weights = has_weight_zero
        self.has_bellman_ford_edges_loop = bool(has_v_minus_1 and (has_edge_iteration or self.has_dist_relaxation))
        self.has_negative_cycle_check = has_neg_cycle_check
        self.has_apsp_nested_sources = has_apsp_sources
        self.has_dag_relaxation_loop = has_dag_relax
        self.has_cycle_path_set = bool(has_path_add and has_path_remove)
        self.has_topological_sort_kahn = bool(has_indegree and (self.uses_queue or self.has_queue_while_loop))
        self.has_topological_sort_dfs = bool((self.has_cycle_path_set or self.has_three_color_cycle_detection or self.has_reversed_stack_return) and not has_dag_relax)

        # Disambiguate path.add/path.remove cycle detection from Permutations backtracking:
        if self.has_cycle_path_set and (self.uses_adjacency_structure or self.has_adjacency_iteration):
            self.has_backtracking_undo_pattern = False

    def _scan_bucket_sort_structures(self, tree):
        """
        Extracts structural Bucket Sort signals:
        - List of lists created via comprehension sized n ([[] for _ in range(n)])
        - Index computed from value (multiplication/division of element, int() cast) used to append into bucket
        - Second loop sorting each bucket
        - Final concatenation loop extending buckets into result
        """
        computed_index_names = set()
        bucket_collection_names = set()

        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.ListComp):
                if isinstance(n.value.elt, ast.List) and len(n.value.elt.elts) == 0:
                    for gen in n.value.generators:
                        if isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range":
                            is_graph_adj = any(
                                isinstance(t, ast.Name) and any(k in t.id.lower() for k in ("adj", "graph", "edge", "matrix", "network"))
                                for t in n.targets
                            )
                            if not is_graph_adj:
                                self.has_bucket_comprehension = True
                                for t in n.targets:
                                    if isinstance(t, ast.Name):
                                        bucket_collection_names.add(t.id)

            if isinstance(n, ast.Assign):
                is_index_calc = False
                if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "int":
                    if any(isinstance(a, (ast.BinOp, ast.Call)) for a in n.value.args):
                        is_index_calc = True
                elif isinstance(n.value, ast.BinOp) and isinstance(n.value.op, (ast.Mult, ast.FloorDiv, ast.Div)):
                    is_index_calc = True
                if is_index_calc:
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            computed_index_names.add(t.id)

            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append":
                if isinstance(n.func.value, ast.Subscript):
                    sl = n.func.value.slice
                    sl_is_computed = (isinstance(sl, ast.Name) and sl.id in computed_index_names) or isinstance(sl, (ast.Call, ast.BinOp))
                    if sl_is_computed:
                        self.has_bucket_index_calculation = True
                    elif isinstance(n.func.value.value, ast.Name) and n.func.value.value.id in bucket_collection_names:
                        # Append to a recognized bucket collection
                        self.has_bucket_index_calculation = True

            if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                tname = n.target.id
                has_sort = any(
                    isinstance(c, ast.Call) and (
                        (isinstance(c.func, ast.Attribute) and c.func.attr in ("sort", "insertion_sort") and isinstance(c.func.value, ast.Name) and c.func.value.id == tname) or
                        (isinstance(c.func, ast.Name) and "sort" in c.func.id.lower() and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == tname)
                    )
                    for c in ast.walk(n)
                )
                if has_sort:
                    self.has_bucket_sort_loop = True
                    self.uses_sorted_or_sort_call = True

            if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                tname = n.target.id
                has_concat = any(
                    (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "extend" and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == tname) or
                    (isinstance(c, ast.AugAssign) and isinstance(c.op, ast.Add) and isinstance(c.value, ast.Name) and c.value.id == tname) or
                    (isinstance(c, ast.For) and isinstance(c.iter, ast.Name) and c.iter.id == tname and any(isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "append" for inner in ast.walk(c)))
                    for c in ast.walk(n)
                )
                if has_concat:
                    self.has_bucket_concatenation = True

    def _scan_prim_mst_structures(self, tree):
        """
        Extracts structural Prim's MST / Dijkstra / Priority Queue traversal signals:
        - has_priority_queue_traversal: While loop popping from min-heap (heappop)
        - has_weighted_adjacency: Iteration over graph with weight unpacking ((v, w) or (w, v))
        - has_mst_weight_accumulation: Scalar accumulation into total_weight / cost / mst_weight
        - has_mst_edge_building: Appending edge tuples to MST edge list (mst.append((u, v)))
        - has_lazy_deletion_on_pop: Skipping already-visited vertices immediately after pop (if v in visited: continue)
        - has_dist_relaxation: Checking or assigning relaxation (dist[v] > d + w, dist[v] = d + w)
        """
        for n in ast.walk(tree):
            # 1. Priority queue traversal in while loop & 5. Lazy deletion on pop
            if isinstance(n, ast.While):
                has_heappop = any(
                    isinstance(c, ast.Call) and (
                        (isinstance(c.func, ast.Name) and c.func.id == "heappop") or
                        (isinstance(c.func, ast.Attribute) and (
                            c.func.attr == "heappop"
                            or (c.func.attr == "get" and any(k in getattr(c.func.value, "id", "").lower() for k in ("pq", "heap", "priority")))
                        ))
                    )
                    for c in ast.walk(n)
                )
                if has_heappop:
                    self.has_priority_queue_traversal = True
                    self.uses_priority_queue = True

                for sub in ast.walk(n):
                    if isinstance(sub, ast.If):
                        is_in_check = (
                            isinstance(sub.test, ast.Compare)
                            and any(isinstance(op, ast.In) for op in sub.test.ops)
                        )
                        has_continue = any(isinstance(s, ast.Continue) for s in sub.body)
                        if is_in_check and has_continue:
                            self.has_lazy_deletion_on_pop = True

            # 2. Weighted adjacency iteration
            if isinstance(n, ast.For) and isinstance(n.target, (ast.Tuple, ast.List)):
                if len(n.target.elts) in (2, 3):
                    iter_is_adj = False
                    if isinstance(n.iter, ast.Subscript) and isinstance(n.iter.value, ast.Name):
                        if any(k in n.iter.value.id.lower() for k in ("graph", "adj", "edges", "network", "g", "node")):
                            iter_is_adj = True
                    elif isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute):
                        if n.iter.func.attr in ("get", "neighbors", "items", "values"):
                            iter_is_adj = True
                    elif self.uses_adjacency_structure:
                        iter_is_adj = True
                    if iter_is_adj:
                        self.has_weighted_adjacency = True

            # 3. MST weight accumulation
            if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add):
                if isinstance(n.target, ast.Name):
                    tname = n.target.id.lower()
                    if any(k in tname for k in ("weight", "cost", "total", "mst", "ans", "res", "sum")):
                        self.has_mst_weight_accumulation = True

            # 4. MST edge building
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append":
                if n.args and isinstance(n.args[0], (ast.Tuple, ast.List)) and len(n.args[0].elts) in (2, 3):
                    if isinstance(n.func.value, ast.Name):
                        lname = n.func.value.id.lower()
                        if any(k in lname for k in ("mst", "tree", "edge", "res", "result")):
                            self.has_mst_edge_building = True

            # 6. Distance relaxation (Dijkstra)
            if isinstance(n, ast.Compare):
                is_dist_comp = any(
                    isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name) and any(k in item.value.id.lower() for k in ("dist", "distance", "dists"))
                    for item in [n.left] + n.comparators
                )
                has_add = any(
                    isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add)
                    for item in ast.walk(n)
                )
                if is_dist_comp and has_add:
                    self.has_dist_relaxation = True

            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        if any(k in target.value.id.lower() for k in ("dist", "distance", "dists")):
                            if isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add):
                                self.has_dist_relaxation = True

    def _scan_binary_search_structure(self, tree):
        """
        High-priority pre-scan for canonical iterative binary search.

        Detects the four structural signals together so that they cannot
        be confused with a general two-pointer or merge-sort pattern:

          1. while <lo> <= <hi>:          → single-path search condition
          2. mid = (<lo> + <hi>) // 2     → midpoint calculation
          3. arr[mid] == <target>          → target equality comparison
          4. <lo> = mid + 1 / <hi> = mid - 1  → directional pointer adjustment

        Setting all three features atomically here means the rule-based
        complexity estimator and the ML classifier both see an unambiguous
        signal regardless of how the loop-level visitors run.
        """
        _LO_NAMES  = {"lo", "left", "l", "low", "start", "begin"}
        _HI_NAMES  = {"hi", "right", "r", "high", "end", "finish"}
        _MID_NAMES = {"mid", "middle", "center", "m"}
        _TGT_NAMES = {"target", "x", "key", "search", "val", "value", "query", "k"}

        for while_node in ast.walk(tree):
            if not isinstance(while_node, ast.While):
                continue

            # ── Signal 1: while lo <= hi ──────────────────────────────────────
            test = while_node.test
            if not isinstance(test, ast.Compare):
                continue
            lo_node = test.left
            if not isinstance(lo_node, ast.Name):
                continue
            if lo_node.id.lower() not in _LO_NAMES:
                continue
            comparators = test.comparators
            if not comparators or not isinstance(comparators[0], ast.Name):
                continue
            if comparators[0].id.lower() not in _HI_NAMES:
                continue
            has_single_path = True

            # ── Signals 2, 3, 4: scan the loop body ──────────────────────────
            has_midpoint = False
            has_target_cmp = False
            has_dir_adjust = False

            for n in ast.walk(ast.Module(body=while_node.body, type_ignores=[])):
                # Signal 2: mid = (lo + hi) // 2   or   mid = lo + (hi - lo) // 2
                if isinstance(n, ast.Assign):
                    for tgt in n.targets:
                        if (
                            isinstance(tgt, ast.Name)
                            and tgt.id.lower() in _MID_NAMES
                            and isinstance(n.value, ast.BinOp)
                            and isinstance(n.value.op, (ast.FloorDiv, ast.RShift, ast.Add))
                        ):
                            has_midpoint = True

                # Signal 3: arr[mid] == target
                if isinstance(n, ast.Compare):
                    for side in [n.left, *n.comparators]:
                        if isinstance(side, ast.Subscript):
                            sl = side.slice
                            if isinstance(sl, ast.Name) and sl.id.lower() in _MID_NAMES:
                                other = n.comparators[0] if side is n.left else n.left
                                if isinstance(other, ast.Name) and other.id.lower() in _TGT_NAMES:
                                    has_target_cmp = True

                # Signal 4: lo = mid + 1  or  hi = mid - 1
                if isinstance(n, ast.Assign):
                    for tgt in n.targets:
                        if isinstance(tgt, ast.Name) and tgt.id.lower() in (_LO_NAMES | _HI_NAMES):
                            val = n.value
                            if isinstance(val, ast.BinOp) and isinstance(val.op, (ast.Add, ast.Sub)):
                                if isinstance(val.left, ast.Name) and val.left.id.lower() in _MID_NAMES:
                                    if isinstance(val.right, ast.Constant) and val.right.value == 1:
                                        has_dir_adjust = True

            # All four signals present → binary search confirmed
            if has_single_path and has_midpoint and has_target_cmp and has_dir_adjust:
                self.has_target_comparison = True
                self.has_directional_pointer_adjustment = True
                self.has_single_path_search = True
                self.has_midpoint_calculation = True  # also set by visit_Assign, but set here too
                return  # short-circuit: no need to scan more While loops

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
        Covers both Kahn's algorithm (in-degree array + zero-in-degree queue)
        and DFS post-order ordering with cycle detection (3-color/in-stack states).
        """
        # 1. Kahn's Algorithm
        has_indegree_var = any(
            isinstance(n, ast.Name) and "indegree" in n.id.lower().replace("_", "")
            for n in ast.walk(tree)
        )
        has_queue_usage = any(
            (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "deque") or
            (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (
                n.func.attr == "popleft" or (n.func.attr == "pop" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 0)
            ))
            for n in ast.walk(tree)
        )
        if has_indegree_var and (has_queue_usage or self.uses_queue or self.has_queue_while_loop):
            self.has_topological_sort_pattern = True

        # 2. 3-Color / Cycle detection (0: unvisited, 1: visiting/in_stack, 2: visited)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        if any(k in target.value.id.lower() for k in ("visited", "seen", "state", "color", "mark")):
                            if isinstance(n.value, ast.Constant) and n.value.value in (1, 2, "visiting", "visited", "gray", "black"):
                                self.has_three_color_cycle_detection = True
            elif isinstance(n, ast.Compare):
                # e.g., if visited[neighbor] == 1: return True
                if isinstance(n.left, ast.Subscript) and isinstance(n.left.value, ast.Name):
                    if any(k in n.left.value.id.lower() for k in ("visited", "seen", "state", "color")):
                        for comp in n.comparators:
                            if isinstance(comp, ast.Constant) and comp.value in (1, 2, "visiting", "visited", "gray", "black"):
                                self.has_three_color_cycle_detection = True

        # 3. Post-order stack append
        # Appending node to a stack or order list after traversing neighbors
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("append", "push"):
                if isinstance(n.func.value, ast.Name):
                    name = n.func.value.id.lower()
                    if any(k in name for k in ("stack", "topo_order", "topo")):
                        self.has_topo_dfs_postorder = True
                    elif (self.uses_adjacency_structure or self.has_adjacency_iteration) and any(k in name for k in ("order", "res", "ans", "seq")):
                        self.has_topo_dfs_postorder = True

        # 4. Reversed stack return: return stack[::-1] or reversed(stack)
        for n in ast.walk(tree):
            if isinstance(n, ast.Return) and n.value:
                if isinstance(n.value, ast.Subscript):
                    sl = n.value.slice
                    if isinstance(sl, ast.Slice) and isinstance(sl.step, ast.UnaryOp) and isinstance(sl.step.op, ast.USub):
                        if isinstance(sl.step.operand, ast.Constant) and sl.step.operand.value == 1:
                            self.has_reversed_stack_return = True
                elif isinstance(n.value, ast.Call):
                    callee = n.value.func.id if isinstance(n.value.func, ast.Name) else ""
                    if callee == "reversed":
                        self.has_reversed_stack_return = True

        # If DFS post-order append + reversed stack return (or 3-color cycle detection) are present with adjacency:
        if self.has_topo_dfs_postorder and (self.has_reversed_stack_return or self.has_three_color_cycle_detection):
            self.has_topological_sort_pattern = True

    def _scan_interval_scheduling_structures(self, tree):
        """
        Extracts greedy interval scheduling signals:
        - sort by end/start time (key=lambda x: x[1] or x[0])
        - single-pass compatibility check: start >= last_end or start > prev_end
        - updating last_end = end and keeping/counting intervals
        - no 2D DP table, no memo dict
        """
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                for kw in n.keywords:
                    if kw.arg == "key" and isinstance(kw.value, ast.Lambda):
                        if isinstance(kw.value.body, ast.Subscript) and isinstance(kw.value.body.slice, ast.Constant):
                            if kw.value.body.slice.value in (0, 1):
                                self.has_interval_sort_key = True

            if isinstance(n, ast.Compare):
                for op in n.ops:
                    if isinstance(op, (ast.GtE, ast.Gt, ast.LtE, ast.Lt)):
                        self.has_interval_compatibility_check = True

        if self.has_interval_sort_key and self.has_interval_compatibility_check:
            self.has_interval_scheduling_pattern = True

    def _scan_counting_sort_structures(self, tree):
        """
        Extracts genuine Counting Sort signals:
        - has_element_indexed_count: count array indexed by element value and incremented (count[x] += 1 or count[arr[i]] += 1)
        - has_count_sort_rebuild: output reconstruction loop from count frequencies or prefix-sums
        """
        for n in ast.walk(tree):
            # 1. Element-indexed frequency increment: count[x] += 1 or count[arr[i]] += 1
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript) and isinstance(n.op, ast.Add):
                if isinstance(n.value, ast.Constant) and n.value.value == 1:
                    arr_name = getattr(n.target.value, "id", "").lower()
                    if any(k in arr_name for k in ("count", "cnt", "freq", "frequency", "bucket", "c")):
                        self.has_element_indexed_count = True
                    if isinstance(n.target.slice, (ast.Name, ast.Subscript)):
                        self.has_element_indexed_count = True

            # 2. Count reconstruction or prefix sum: count[i] += count[i-1] or output[count[x]-1] = x
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript) and isinstance(n.value, ast.Subscript):
                if getattr(n.target.value, "id", "") == getattr(n.value.value, "id", ""):
                    self.has_count_sort_rebuild = True
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.slice, (ast.BinOp, ast.Subscript)):
                        arr_name = getattr(t.value, "id", "").lower()
                        if any(k in arr_name for k in ("output", "res", "ans", "sorted", "b")):
                            self.has_count_sort_rebuild = True

    def _scan_word_break_structures(self, tree):
        """
        Extracts Word Break / string partition DP signals:
        - has_string_slicing_subscript: string slicing s[j:i] or s[:i]
        - has_earlier_dp_dependency: dp[j] read inside inner loop j < i where dp[i] is updated
        - has_word_break_pattern: dp earlier dependency + string slice membership in dictionary/set
        """
        for n in ast.walk(tree):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
                self.has_string_slicing_subscript = True

        for n in ast.walk(tree):
            if isinstance(n, (ast.For, ast.While)):
                for inner in ast.walk(n):
                    if isinstance(inner, (ast.For, ast.While)):
                        for cond in ast.walk(inner):
                            if isinstance(cond, ast.Compare):
                                is_in_word = any(
                                    any(w in getattr(c, "id", "").lower() for w in ("word", "dict", "words", "set", "vocab"))
                                    for c in cond.comparators if isinstance(c, ast.Name)
                                ) or any(
                                    isinstance(op, ast.In) for op in cond.ops
                                )
                                has_slice_left = isinstance(cond.left, ast.Subscript) and isinstance(cond.left.slice, ast.Slice)
                                if is_in_word and has_slice_left:
                                    self.has_word_break_pattern = True

                            if isinstance(cond, ast.BoolOp) and isinstance(cond.op, ast.And):
                                for val in cond.values:
                                    if isinstance(val, ast.Subscript) and isinstance(val.slice, ast.Name):
                                        tname = getattr(val.value, "id", "").lower()
                                        if any(k in tname for k in ("dp", "memo", "table", "valid", "can")):
                                            self.has_earlier_dp_dependency = True

        if self.has_earlier_dp_dependency and (self.has_string_slicing_subscript or self.has_word_break_pattern):
            self.has_word_break_pattern = True

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
        # Note: full shallow copy arr[:] has neither lower nor upper bound, so it is not a split slice
        if any(
            isinstance(arg, ast.Subscript)
            and isinstance(arg.slice, ast.Slice)
            and (arg.slice.lower is not None or arg.slice.upper is not None)
            for arg in node.args
        ):
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
        self._check_single_path_search(node)
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

    def _check_single_path_search(self, node):
        """Detects Binary Search single-path execution: while left <= right with no merging"""
        # Binary Search typically has a while loop with left <= right condition
        # and does NOT have auxiliary merging (which Merge Sort does)
        if isinstance(node.test, ast.Compare):
            left = node.test.left
            comparators = node.test.comparators
            # Check for left <= right pattern
            if isinstance(left, ast.Name) and left.id.lower() in ("left", "lo", "l", "low"):
                if comparators and isinstance(comparators[0], ast.Name):
                    right_name = comparators[0].id.lower()
                    if right_name in ("right", "hi", "r", "high"):
                        # This is a two-pointer while loop, likely Binary Search
                        # Check if there's no auxiliary merging in the body
                        has_merging = False
                        for n in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                                if n.func.attr in ("append", "extend"):
                                    has_merging = True
                                    break
                        if not has_merging and not self.has_auxiliary_merging:
                            self.has_single_path_search = True

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
        assign_idx, assign_target = None, None

        current_fn = self.current_function_stack[-1] if self.current_function_stack else None

        for idx, stmt in enumerate(body):
            # Check if this statement triggers recursion
            has_recurse = False
            if current_fn:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Call):
                        if (isinstance(sub.func, ast.Name) and sub.func.id == current_fn) or \
                           (isinstance(sub.func, ast.Attribute) and sub.func.attr == current_fn):
                            has_recurse = True
                            break
            if has_recurse:
                recurse_idx = idx

            # Method-based undo: path.append(...) ... recurse ... path.pop()
            call = stmt.value if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) else None
            if call is not None:
                if isinstance(call.func, ast.Attribute) and call.func.attr in add_methods:
                    if isinstance(call.func.value, ast.Name):
                        add_idx, add_var = idx, call.func.value.id

                elif isinstance(call.func, ast.Attribute) and call.func.attr in remove_methods:
                    if (
                        isinstance(call.func.value, ast.Name)
                        and call.func.value.id == add_var
                        and add_idx is not None
                        and recurse_idx is not None
                        and add_idx < recurse_idx < idx
                    ):
                        self.has_backtracking_undo_pattern = True

            # Subscript / variable assignment undo:
            # board[r][c] = 'Q' ... recurse ... board[r][c] = '.'
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, (ast.Subscript, ast.Name)):
                        target_repr = ast.dump(target)
                        if assign_idx is None:
                            assign_idx, assign_target = idx, target_repr
                        elif assign_idx is not None and recurse_idx is not None and assign_idx < recurse_idx <= idx:
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
        self._check_target_comparison(node)
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
        self._check_directional_pointer_adjustment(node)
        self.generic_visit(node)

    def _check_gap_assign(self, node):
        """gap = n // 2 or gap = int(gap / 1.3)"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id.lower()
            if "gap" in name or "shrink" in name:
                self.has_gap_reduction = True

    def _check_directional_pointer_adjustment(self, node):
        """Detects Binary Search pointer adjustments: left = mid + 1, right = mid - 1"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id.lower()
            # Check for left/right pointer variable names
            if target_name in ("left", "right", "lo", "hi", "low", "high", "l", "r"):
                if isinstance(node.value, ast.BinOp):
                    # left = mid + 1 or right = mid - 1
                    if isinstance(node.value.op, (ast.Add, ast.Sub)):
                        # Check if the other operand is a constant (usually 1 or -1)
                        if isinstance(node.value.right, ast.Constant):
                            if node.value.right.value in (1, -1):
                                # Check if the left side is a midpoint-like variable
                                if isinstance(node.value.left, ast.Name):
                                    left_name = node.value.left.id.lower()
                                    if any(k in left_name for k in ("mid", "middle", "pivot", "center")):
                                        self.has_directional_pointer_adjustment = True

    def _check_target_comparison(self, node):
        """Detects Binary Search target comparison: arr[mid] == target or arr[mid] == x"""
        if isinstance(node.test, ast.Compare):
            # Check for equality comparison (arr[mid] == target)
            if isinstance(node.test.ops[0], ast.Eq):
                left = node.test.left
                comparators = node.test.comparators
                # Check if one side is array[mid] and other is target-like variable
                for side in [left, *comparators]:
                    if isinstance(side, ast.Subscript):
                        # Check if it's array[mid] pattern
                        if isinstance(side.slice, ast.Name):
                            slice_name = side.slice.id.lower()
                            if any(k in slice_name for k in ("mid", "middle", "center")):
                                # Found array[mid], now check the other side for target
                                other_side = comparators[0] if side == left else left
                                if isinstance(other_side, ast.Name):
                                    other_name = other_side.id.lower()
                                    if any(k in other_name for k in ("target", "x", "key", "search", "val", "value")):
                                        self.has_target_comparison = True

    def _check_frequency_array(self, node):
        """Pattern: count = [0] * span, count = [0] * 10, holes = [[] for _ in range(...)]"""
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            target_name = node.targets[0].id.lower() if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else ""
            if any(k in target_name for k in ("dp", "memo", "table", "visited", "seen", "valid", "can")):
                return
            mult_val = node.value.right if isinstance(node.value.left, ast.List) else (node.value.left if isinstance(node.value.right, ast.List) else None)
            is_zero_list = False
            if isinstance(node.value.left, ast.List) and len(node.value.left.elts) == 1 and isinstance(node.value.left.elts[0], ast.Constant) and type(node.value.left.elts[0].value) is int and node.value.left.elts[0].value == 0:
                is_zero_list = True
            elif isinstance(node.value.right, ast.List) and len(node.value.right.elts) == 1 and isinstance(node.value.right.elts[0], ast.Constant) and type(node.value.right.elts[0].value) is int and node.value.right.elts[0].value == 0:
                is_zero_list = True

            if is_zero_list:
                if any(k in target_name for k in ("count", "cnt", "freq", "bucket", "hole", "radix", "span")):
                    self.has_frequency_array = True
                elif isinstance(mult_val, ast.Constant) and mult_val.value in (10, 256, 128, 16):
                    self.has_frequency_array = True
                elif isinstance(mult_val, ast.Name) and any(k in mult_val.id.lower() for k in ("span", "range", "max", "k")):
                    self.has_frequency_array = True
                elif isinstance(mult_val, ast.BinOp) and any(k in target_name for k in ("count", "cnt", "freq", "bucket")):
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
        """Detects child math: 2*i + 1, 2*i + 2, 2*idx + 1, (i << 1) + 1, etc."""
        if isinstance(node.op, ast.Add):
            left, right = node.left, node.right
            if isinstance(right, ast.Constant) and right.value in (1, 2):
                if isinstance(left, ast.BinOp) and isinstance(left.op, (ast.Mult, ast.LShift)):
                    if (isinstance(left.left, ast.Constant) and left.left.value in (1, 2)) or \
                       (isinstance(left.right, ast.Constant) and left.right.value in (1, 2)):
                        self.has_heap_child_math = True
            elif isinstance(left, ast.Constant) and left.value in (1, 2):
                if isinstance(right, ast.BinOp) and isinstance(right.op, (ast.Mult, ast.LShift)):
                    if (isinstance(right.left, ast.Constant) and right.left.value in (1, 2)) or \
                       (isinstance(right.right, ast.Constant) and right.right.value in (1, 2)):
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
        # Search-specific features
        "has_target_comparison": float(visitor.has_target_comparison),
        "has_directional_pointer_adjustment": float(visitor.has_directional_pointer_adjustment),
        "has_single_path_search": float(visitor.has_single_path_search),
        # BFS discriminative features
        "has_queue_while_loop": float(visitor.has_queue_while_loop),
        "has_queue_pop": float(visitor.has_queue_pop),
        "has_visited_check_before_enqueue": float(visitor.has_visited_check_before_enqueue),
        "has_adjacency_iteration": float(visitor.has_adjacency_iteration),
        "has_dp_table_indexing": float(visitor.has_dp_table_indexing),
        "has_memo_dict_access": float(visitor.has_memo_dict_access),
        # Bucket Sort structural features
        "has_bucket_comprehension": float(visitor.has_bucket_comprehension),
        "has_bucket_index_calculation": float(visitor.has_bucket_index_calculation),
        "has_bucket_sort_loop": float(visitor.has_bucket_sort_loop),
        "has_bucket_concatenation": float(visitor.has_bucket_concatenation),
        # Prim / Dijkstra / MST discriminative features
        "has_priority_queue_traversal": float(visitor.has_priority_queue_traversal),
        "has_weighted_adjacency": float(visitor.has_weighted_adjacency),
        "has_mst_weight_accumulation": float(visitor.has_mst_weight_accumulation),
        "has_mst_edge_building": float(visitor.has_mst_edge_building),
        "has_lazy_deletion_on_pop": float(visitor.has_lazy_deletion_on_pop),
        "has_dist_relaxation": float(visitor.has_dist_relaxation),
        # Topological Sort discriminative features
        "has_topo_dfs_postorder": float(visitor.has_topo_dfs_postorder),
        "has_three_color_cycle_detection": float(visitor.has_three_color_cycle_detection),
        "has_reversed_stack_return": float(visitor.has_reversed_stack_return),
        # Greedy / Interval Scheduling discriminative features
        "has_interval_scheduling_pattern": float(visitor.has_interval_scheduling_pattern),
        "has_interval_sort_key": float(visitor.has_interval_sort_key),
        "has_interval_compatibility_check": float(visitor.has_interval_compatibility_check),
        # Word Break vs Counting Sort discriminative features
        "has_element_indexed_count": float(visitor.has_element_indexed_count),
        "has_count_sort_rebuild": float(visitor.has_count_sort_rebuild),
        "has_string_slicing_subscript": float(visitor.has_string_slicing_subscript),
        "has_earlier_dp_dependency": float(visitor.has_earlier_dp_dependency),
        "has_word_break_pattern": float(visitor.has_word_break_pattern),
        # Graph family discriminative features
        "has_reversed_neighbor_traversal": float(visitor.has_reversed_neighbor_traversal),
        "has_stack_extend_neighbors": float(visitor.has_stack_extend_neighbors),
        "has_deque_appendleft": float(visitor.has_deque_appendleft),
        "has_bellman_ford_edges_loop": float(visitor.has_bellman_ford_edges_loop),
        "has_negative_cycle_check": float(visitor.has_negative_cycle_check),
        "has_apsp_nested_sources": float(visitor.has_apsp_nested_sources),
        "has_dag_relaxation_loop": float(visitor.has_dag_relaxation_loop),
        "has_cycle_path_set": float(visitor.has_cycle_path_set),
        "has_zero_one_weights": float(visitor.has_zero_one_weights),
        "has_topological_sort_kahn": float(visitor.has_topological_sort_kahn),
        "has_topological_sort_dfs": float(visitor.has_topological_sort_dfs),
    }

    # Extract any top-level parameters (e.g. n = 8 for N-Queens, nums = [9, 8, 7] for Permutations)
    params = extract_top_level_parameters(code)
    if "n" in params:
        features["detected_n_value"] = float(params["n"])
    if "nums" in params:
        features["detected_nums_len"] = float(len(params["nums"]))

    return features


def _parse_ast_list(node: ast.List) -> List[Any]:
    result = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant):
            result.append(elt.value)
        elif isinstance(elt, ast.Num):
            result.append(elt.n)
        elif isinstance(elt, ast.Str):
            result.append(elt.s)
        elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
            if isinstance(elt.operand, ast.Constant):
                result.append(-elt.operand.value)
            elif isinstance(elt.operand, ast.Num):
                result.append(-elt.operand.n)
    return result


def _parse_raw_list_str(s: str) -> List[Any]:
    parts = [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
    if parts and all(re.match(r"^-?\d+$", p) for p in parts):
        return [int(p) for p in parts]
    return parts


def extract_top_level_parameters(code: str) -> Dict[str, Any]:
    """
    Extracts top-level variable assignments or algorithm call arguments from source code,
    such as `n = 8` or `solve_n_queens(8)` or `permute([9, 8, 7])`.
    Supports commented invocations like `# permute([9,8,7])`.
    """
    params: Dict[str, Any] = {}
    try:
        tree = ast.parse(code)
    except Exception:
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                            params[var_name] = node.value.value
                        elif isinstance(node.value, ast.Num):
                            params[var_name] = node.value.n
                        elif isinstance(node.value, ast.List):
                            if var_name in ("nums", "arr", "items", "candidates", "elements"):
                                params["nums"] = _parse_ast_list(node.value)
                            else:
                                params[var_name] = _parse_ast_list(node.value)
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr.lower()

                if any(k in func_name for k in ("queen", "nqueens", "n_queens")):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                            params["n"] = int(arg.value)
                        elif isinstance(arg, ast.Num):
                            params["n"] = int(arg.n)
                elif any(k in func_name for k in ("permute", "permutations")):
                    for arg in node.args:
                        if isinstance(arg, ast.List):
                            params["nums"] = _parse_ast_list(arg)
                    for kw in node.keywords:
                        if kw.arg in ("nums", "arr") and isinstance(kw.value, ast.List):
                            params["nums"] = _parse_ast_list(kw.value)

    # Regex fallbacks (for comments like `# permute([9,8,7])` or syntax errors)
    if "n" not in params:
        m = re.search(r"(?:^|\n)\s*(?:board_size|boardSize|num_queens|queens|\bn\b|\bN\b)\s*=\s*(\d+)", code, re.IGNORECASE)
        if m:
            params["n"] = int(m.group(1))
        m2 = re.search(r"(?:solve_?n_?queens|nqueens|n_queens|queens)\s*\(\s*(\d+)\s*\)", code, re.IGNORECASE)
        if m2:
            params["n"] = int(m2.group(1))

    if "nums" not in params:
        m_perm = re.search(r"(?:permute|permutations)\s*\(\s*(?:nums\s*=\s*)?\[([^\]]+)\]", code, re.IGNORECASE)
        if m_perm:
            params["nums"] = _parse_raw_list_str(m_perm.group(1))
        else:
            m_arr = re.search(r"(?:^|\n)\s*(?:nums|arr|items|candidates)\s*=\s*\[([^\]]+)\]", code, re.IGNORECASE)
            if m_arr:
                params["nums"] = _parse_raw_list_str(m_arr.group(1))

    return params


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