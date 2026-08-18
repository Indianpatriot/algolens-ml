"""
Safe-ish execution tracer for user-submitted Python snippets.

The public entry point is trace_execution(code, entry_call=None). It runs the
submitted code in a separate process, traces line/call/return events from that
code only, and returns JSON-serializable execution snapshots.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import heapq
import io
import multiprocessing
import sys
import traceback
from types import FrameType
from typing import Any


USER_FILENAME = "<algolens_user_code>"
MAX_STEPS = 500
MAX_REPR_LENGTH = 1000
TIMEOUT_SECONDS = 5
MEMORY_LIMIT_BYTES = 128 * 1024 * 1024

BLOCKED_NAMES = {
    "open",
    "__import__",
    "exec",
    "eval",
    "compile",
    "input",
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
}

ALLOWED_IMPORTS = {"collections", "heapq"}


class TraceLimitExceeded(BaseException):
    pass


def trace_execution(code: str, entry_call: str | None = None) -> dict:
    """Trace submitted Python code in an isolated subprocess."""
    if not code or not code.strip():
        return {"steps": [], "final_output": "", "error": "`code` field is empty."}

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_trace_worker, args=(code, entry_call, queue))
    process.start()
    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        return {
            "steps": [],
            "final_output": "",
            "error": "Execution timed out - possible infinite loop",
        }

    try:
        return queue.get_nowait()
    except Exception:
        return {
            "steps": [],
            "final_output": "",
            "error": "Execution failed before a trace result could be returned.",
        }


def _trace_worker(code: str, entry_call: str | None, queue: multiprocessing.Queue) -> None:
    _apply_memory_limit()

    stdout = io.StringIO()
    steps: list[dict[str, Any]] = []
    try:
        prepared_code = _prepare_code(code, entry_call)
        source_lines = prepared_code.splitlines()
        compiled = compile(prepared_code, USER_FILENAME, "exec")
    except SyntaxError as exc:
        queue.put(
            {
                "steps": [],
                "final_output": "",
                "error": f"Syntax error on line {exc.lineno}: {exc.msg}",
            }
        )
        return
    except ValueError as exc:
        queue.put({"steps": [], "final_output": "", "error": str(exc)})
        return
    except Exception as exc:
        queue.put({"steps": [], "final_output": "", "error": f"Could not prepare code: {exc}"})
        return

    globals_ns = _safe_globals()
    call_depth = 0
    truncated = False

    def tracer(frame: FrameType, event: str, arg: Any):
        nonlocal call_depth, truncated

        if frame.f_code.co_filename != USER_FILENAME:
            return tracer
        if frame.f_code.co_name == "<module>" and event in {"call", "return"}:
            return tracer

        if event == "call":
            call_depth += 1

        if event in {"call", "line", "return"}:
            if len(steps) >= MAX_STEPS:
                truncated = True
                raise TraceLimitExceeded()
            steps.append(_snapshot(frame, event, source_lines, call_depth))

        if event == "return":
            call_depth = max(0, call_depth - 1)

        return tracer

    try:
        with contextlib.redirect_stdout(stdout):
            sys.settrace(tracer)
            exec(compiled, globals_ns, globals_ns)
    except TraceLimitExceeded:
        pass
    except Exception:
        error = traceback.format_exc(limit=1).strip()
        sys.settrace(None)
        queue.put({"steps": steps, "final_output": stdout.getvalue(), "error": error})
        return
    finally:
        sys.settrace(None)

    error = None
    if truncated:
        error = "Trace truncated at 500 steps (execution may be long-running)."

    queue.put({"steps": steps, "final_output": stdout.getvalue(), "error": error})


def _prepare_code(code: str, entry_call: str | None) -> str:
    tree = ast.parse(code)
    _validate_ast(tree)

    if entry_call and entry_call.strip():
        entry_tree = ast.parse(entry_call, mode="exec")
        _validate_ast(entry_tree)
        return f"{code.rstrip()}\n\n{entry_call.strip()}\n"

    if _only_definitions_and_imports(tree):
        raise ValueError(
            "No executable sample call found. Add a call at the bottom, e.g. "
            "print(binary_search([1, 2, 3], 2))."
        )

    return code


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Use of restricted name '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Access to dunder attributes is not allowed.")


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        module_names = [alias.name.split(".", 1)[0] for alias in node.names]
    else:
        if node.module is None:
            raise ValueError("Relative imports are not allowed.")
        module_names = [node.module.split(".", 1)[0]]

    for module_name in module_names:
        if module_name not in ALLOWED_IMPORTS:
            raise ValueError(
                f"Import of '{module_name}' is not allowed. Only collections and heapq are available."
            )


def _only_definitions_and_imports(tree: ast.Module) -> bool:
    saw_function_or_class = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            saw_function_or_class = True
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        return False
    return saw_function_or_class


def _safe_globals() -> dict[str, Any]:
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": _safe_import,
        "__build_class__": __build_class__,
        "object": object,
    }
    return {
        "__builtins__": safe_builtins,
        "__name__": "__algolens_trace__",
        "collections": collections,
        "heapq": heapq,
    }


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".", 1)[0]
    if level != 0 or root_name not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed.")
    if root_name == "collections":
        return collections
    if root_name == "heapq":
        return heapq
    raise ImportError(f"Import of '{name}' is not allowed.")


def _snapshot(
    frame: FrameType,
    event: str,
    source_lines: list[str],
    call_depth: int,
) -> dict[str, Any]:
    line_number = frame.f_lineno
    line_text = ""
    if 1 <= line_number <= len(source_lines):
        line_text = source_lines[line_number - 1]

    snapshot: dict[str, Any] = {
        "line_number": line_number,
        "line_text": line_text,
        "locals": _safe_locals(frame.f_locals),
        "call_depth": call_depth,
        "event": event,
    }

    op_info = _detect_ds_operation(line_text, frame.f_locals)
    if op_info:
        snapshot["operation"] = op_info.get("operation")
        snapshot["ds_type"] = op_info.get("type")
        if "data" in op_info:
            snapshot["data"] = op_info["data"]
        if "active_indices" in op_info:
            snapshot["active_indices"] = op_info["active_indices"]
        if "pointer_indices" in op_info:
            snapshot["pointer_indices"] = op_info["pointer_indices"]
        if "log" in op_info:
            snapshot["log"] = op_info["log"]

    return snapshot


def _detect_ds_operation(line_text: str, locals_dict: dict[str, Any]) -> dict[str, Any] | None:
    import re as _re

    trimmed = line_text.strip()
    if not trimmed:
        return None

    # ------------------------------------------------------------------ #
    # Helper: extract numeric/string argument from a method call            #
    # e.g.  stack.append(42)  -> "42",  q.append("hello") -> "hello"       #
    # ------------------------------------------------------------------ #
    def _extract_arg(text: str) -> Any:
        m = _re.search(r'\.\w+\(\s*([^\)]+?)\s*\)', text)
        if not m:
            return None
        raw = m.group(1).strip().strip("'\"")
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        # Try to resolve variable name from locals
        if raw in locals_dict:
            return locals_dict[raw]
        return raw

    # ------------------------------------------------------------------ #
    # Queue dequeue: .popleft() or .pop(0)                                  #
    # ------------------------------------------------------------------ #
    if ".popleft(" in trimmed or ".pop(0)" in trimmed:
        for k, v in locals_dict.items():
            if isinstance(v, (list, collections.deque)):
                data = list(v)
                front_val = data[0] if data else None
                return {
                    "operation": "dequeue",
                    "type": "queue",
                    "value": front_val,
                    "data": data[:20],
                    "active_indices": [0] if data else [],
                    "pointer_indices": {"front": 0 if data else -1, "rear": len(data) - 1 if data else -1},
                    "log": f"Dequeued {front_val} from queue. Queue is now: {data[1:20]}",
                }
        return {"operation": "dequeue", "type": "queue", "log": "Dequeued from queue."}

    # ------------------------------------------------------------------ #
    # Stack pop: .pop() with no argument (distinguish from .pop(0))        #
    # ------------------------------------------------------------------ #
    if _re.search(r'\.pop\(\s*\)', trimmed) and ".pop(0)" not in trimmed:
        for k, v in locals_dict.items():
            if isinstance(v, list) and ("stack" in k or "stk" in k or "arr" in k.lower() or "s" == k):
                data = list(v)
                top = len(data) - 1
                top_val = data[top] if top >= 0 else None
                return {
                    "operation": "pop",
                    "type": "stack",
                    "value": top_val,
                    "data": data[:20],
                    "active_indices": [top] if top >= 0 else [],
                    "pointer_indices": {"top": top},
                    "log": f"Popped {top_val} from stack. Stack is now: {data[:top]}",
                }
        # Fall back to any list variable
        for k, v in locals_dict.items():
            if isinstance(v, list) and k not in {"result", "res", "output", "ans"}:
                data = list(v)
                top = len(data) - 1
                top_val = data[top] if top >= 0 else None
                return {
                    "operation": "pop",
                    "type": "stack",
                    "value": top_val,
                    "data": data[:20],
                    "active_indices": [top] if top >= 0 else [],
                    "pointer_indices": {"top": top},
                    "log": f"Popped {top_val} from stack. Stack is now: {data[:top]}",
                }
        return {"operation": "pop", "type": "stack", "log": "Popped from stack."}

    # ------------------------------------------------------------------ #
    # Queue enqueue: .append() on a queue/deque variable                   #
    # ------------------------------------------------------------------ #
    if (
        _re.search(r'\b(queue|deque|q|frontier)\b', trimmed)
        and ".append(" in trimmed
    ):
        pushed_val = _extract_arg(trimmed)
        for k, v in locals_dict.items():
            if isinstance(v, (list, collections.deque)):
                data = list(v)
                return {
                    "operation": "enqueue",
                    "type": "queue",
                    "value": pushed_val,
                    "data": data[:20],
                    "active_indices": [len(data) - 1] if data else [],
                    "pointer_indices": {"front": 0 if data else -1, "rear": len(data) - 1 if data else -1},
                    "log": f"Enqueued {pushed_val} to queue. Queue is now: {data[:20]}",
                }
        return {"operation": "enqueue", "type": "queue", "value": pushed_val, "log": f"Enqueued {pushed_val} to queue."}

    # ------------------------------------------------------------------ #
    # Stack push: .append() or .push() on a stack variable                 #
    # ------------------------------------------------------------------ #
    if (
        _re.search(r'\b(stack|stk)\b', trimmed)
        and _re.search(r'\.(append|push)\(', trimmed)
    ):
        pushed_val = _extract_arg(trimmed)
        for k, v in locals_dict.items():
            if isinstance(v, list) and ("stack" in k or "stk" in k):
                data = list(v)
                top = len(data) - 1
                return {
                    "operation": "push",
                    "type": "stack",
                    "value": pushed_val,
                    "data": data[:20],
                    "active_indices": [top] if top >= 0 else [],
                    "pointer_indices": {"top": top},
                    "log": f"Pushed {pushed_val} to stack. Stack is now: {data[:20]}",
                }
        return {"operation": "push", "type": "stack", "value": pushed_val, "log": f"Pushed {pushed_val} to stack."}

    return None


def _safe_locals(locals_dict: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in locals_dict.items():
        if key in {"__builtins__", "__name__"}:
            continue
        if (key == "collections" and value is collections) or (key == "heapq" and value is heapq):
            continue
        safe[key] = _safe_value(value)
    return safe


def _safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return _truncated_repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return _truncate_string(value) if isinstance(value, str) else value
    if isinstance(value, (list, tuple)):
        if _repr_too_long(value):
            return _truncated_repr(value)
        return [_safe_value(item, depth + 1) for item in value[:50]]
    if isinstance(value, set):
        if _repr_too_long(value):
            return _truncated_repr(value)
        return [_safe_value(item, depth + 1) for item in list(value)[:50]]
    if isinstance(value, dict):
        if _repr_too_long(value):
            return _truncated_repr(value)
        return {
            str(_safe_value(key, depth + 1)): _safe_value(val, depth + 1)
            for key, val in list(value.items())[:50]
        }
    return _truncated_repr(value)


def _repr_too_long(value: Any) -> bool:
    try:
        return len(repr(value)) > MAX_REPR_LENGTH
    except Exception:
        return True


def _truncated_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    return _truncate_string(text)


def _truncate_string(text: str) -> str:
    if len(text) <= MAX_REPR_LENGTH:
        return text
    return f"{text[:MAX_REPR_LENGTH]}... (truncated)"


def _apply_memory_limit() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except Exception:
        pass


__all__ = ["trace_execution"]
