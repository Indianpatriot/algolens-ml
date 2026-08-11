"""
Shared naming pools + rendering helpers for AlgoLens dataset generation.
Each canonical template uses __TOKEN__ placeholders (not .format braces,
to avoid clashing with dict/set literals in the code itself).
"""

import random
import re

# 5 alternate "identifier style" sets per role. Index 0-4 selects a style.
NAME_SETS = [
    dict(ARR="arr", N="n", I="i", J="j", K="k", RESULT="result", TMP="tmp",
         TARGET="target", VISITED="visited", QUEUE="queue", STACK="stack",
         GRAPH="graph", NODE="node", NEIGHBOR="neighbor", LEFT="left",
         RIGHT="right", MID="mid", START="start", END="end", WINDOW_SUM="window_sum",
         COUNT="count", MEMO="memo", DP="dp", PARENT="parent", RANK="rank",
         SEEN="seen", MAXV="max_val", CUR="cur", PREV="prev", DIFF="diff"),
    dict(ARR="nums", N="length", I="idx", J="jdx", K="kth", RESULT="output",
         TMP="swap_val", TARGET="goal", VISITED="explored", QUEUE="frontier",
         STACK="pending", GRAPH="adj_list", NODE="u", NEIGHBOR="v", LEFT="lo",
         RIGHT="hi", MID="mid_pt", START="s", END="e", WINDOW_SUM="cur_sum",
         COUNT="cnt", MEMO="cache", DP="table", PARENT="root_of", RANK="size",
         SEEN="marked", MAXV="best", CUR="curr", PREV="last", DIFF="delta"),
    dict(ARR="data", N="size", I="p1", J="p2", K="pivot_idx", RESULT="ans",
         TMP="hold", TARGET="want", VISITED="vis", QUEUE="q", STACK="stk",
         GRAPH="edges_of", NODE="node_id", NEIGHBOR="nbr", LEFT="left_ptr",
         RIGHT="right_ptr", MID="middle", START="begin", END="finish",
         WINDOW_SUM="running_sum", COUNT="freq", MEMO="lookup", DP="dp_arr",
         PARENT="uf_parent", RANK="uf_rank", SEEN="was_seen", MAXV="max_so_far",
         CUR="c", PREV="p", DIFF="gap"),
    dict(ARR="values", N="num_items", I="row", J="col", K="k_val", RESULT="res",
         TMP="temp", TARGET="needle", VISITED="been_here", QUEUE="bfs_q",
         STACK="dfs_stack", GRAPH="graph_map", NODE="curr_node", NEIGHBOR="adj",
         LEFT="l", RIGHT="r", MID="m", START="start_idx", END="end_idx",
         WINDOW_SUM="win_total", COUNT="counter", MEMO="memo_dict", DP="dp_grid",
         PARENT="par", RANK="rnk", SEEN="checked", MAXV="max_area", CUR="x",
         PREV="prev_x", DIFF="d"),
    dict(ARR="items", N="total", I="a", J="b", K="k_th", RESULT="final_result",
         TMP="t", TARGET="tgt", VISITED="visited_set", QUEUE="to_visit",
         STACK="call_stack", GRAPH="neighbors_of", NODE="w", NEIGHBOR="nxt",
         LEFT="low", RIGHT="high", MID="pivot", START="from_idx", END="to_idx",
         WINDOW_SUM="subarr_sum", COUNT="occurrences", MEMO="memo_cache",
         DP="dp_table", PARENT="find_parent", RANK="tree_rank", SEEN="done",
         MAXV="tallest", CUR="cur_val", PREV="prev_val", DIFF="difference"),
]

FUNC_NAME_POOLS = {
    "bfs": ["bfs", "breadth_first_search", "shortest_path_bfs", "level_order"],
    "dfs": ["dfs", "depth_first_search", "explore", "traverse_dfs"],
    "dp": ["solve", "compute", "dp_solution", "run_dp"],
    "backtracking": ["backtrack", "solve_backtrack", "generate", "search_solutions"],
    "two_pointer": ["two_pointer_solve", "find_pair", "solve_two_ptr", "run"],
    "sliding_window": ["sliding_window_solve", "max_window", "find_window", "solve_window"],
    "binary_search": ["binary_search", "bsearch", "find_target", "search_sorted"],
    "sorting": ["sort_array", "merge_sort", "quick_sort", "sort_solve"],
    "union_find": ["solve", "process_edges", "count_components", "union_find_solve"],
    "monotonic_stack": ["solve_stack", "next_greater", "stack_solve", "process"],
}


def render(template: str, variant: int, comment_on: bool, func_name: str = None) -> str:
    """Fill placeholders using NAME_SETS[variant % 5], optionally strip comments."""
    ns = NAME_SETS[variant % len(NAME_SETS)]
    code = template
    if func_name:
        code = code.replace("__FUNC__", func_name)
    for token, name in ns.items():
        code = code.replace(f"__{token}__", name)
    # any placeholder not covered by an explicit name pool (e.g. __HELPER__,
    # __ROOT_X__) still needs to become a valid, readable identifier
    code = re.sub(r"__([A-Z0-9_]+)__", lambda m: m.group(1).lower(), code)
    if not comment_on:
        lines = [ln for ln in code.split("\n") if not ln.strip().startswith("#")]
        code = "\n".join(lines)
    return code.strip() + "\n"


def pick_func_name(category_key: str, variant: int) -> str:
    pool = FUNC_NAME_POOLS[category_key]
    return pool[variant % len(pool)]