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

    return {
        "line_number": line_number,
        "line_text": line_text,
        "locals": _safe_locals(frame.f_locals),
        "call_depth": call_depth,
        "event": event,
    }


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
