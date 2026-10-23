"""Restricted execution of LLM-written pandas code.

This is *defence in depth for a demo app*, not a hardened sandbox: code is
AST-checked (no imports, no dunder access, no I/O helpers) and run with a
whitelist of builtins. For untrusted multi-tenant use, run it in a separate
container/process with resource limits.
"""
from __future__ import annotations

import ast
import builtins
import logging
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import QueryExecutionError, UnsafeQueryError

log = logging.getLogger(__name__)

_BANNED_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "type", "super", "dir",
}
_BANNED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_parquet", "read_sql",
    "read_table", "read_html", "read_clipboard", "to_csv", "to_excel", "to_json",
    "to_pickle", "to_parquet", "to_sql", "to_clipboard", "to_hdf", "to_feather",
    "eval", "system", "popen", "load", "save", "loadtxt", "genfromtxt", "fromfile", "tofile",
}
_SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in (
        "abs all any bool dict enumerate filter float int isinstance len list map max min "
        "pow range reversed round set slice sorted str sum tuple zip print True False None "
        "Exception ValueError KeyError TypeError"
    ).split()
}


def validate_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise QueryExecutionError(f"Python syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeQueryError("Imports are not allowed; `pd` and `np` are pre-loaded.")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise UnsafeQueryError(f"Use of `{node.id}` is not allowed.")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _BANNED_ATTRS
        ):
            raise UnsafeQueryError(f"Access to `.{node.attr}` is not allowed.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeQueryError("global/nonlocal are not allowed.")
    return tree


@dataclass
class PandasResult:
    df: pd.DataFrame
    code: str


def _to_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index() if not isinstance(value.index, pd.RangeIndex) else value
    if isinstance(value, pd.Series):
        name = value.name if value.name is not None else "value"
        return value.rename(name).reset_index()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return pd.DataFrame({"value": list(value)})
    if isinstance(value, dict):
        return pd.DataFrame([value]) if all(np.isscalar(v) for v in value.values()) else pd.DataFrame(value)
    return pd.DataFrame({"value": [value]})


def run_pandas(code: str, tables: dict[str, pd.DataFrame], timeout_s: float = 20.0,
               max_rows: int = 1000) -> PandasResult:
    """Run `code`; it must assign its answer to a variable named ``result``."""
    tree = validate_code(code)
    env = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np}
    env.update({name: df.copy() for name, df in tables.items()})
    env["dfs"] = {name: env[name] for name in tables}

    outcome: dict[str, object] = {}

    def _target() -> None:
        try:
            exec(compile(tree, "<analysis>", "exec"), env)  # noqa: S102 - validated AST
            outcome["ok"] = True
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM for self-correction
            outcome["err"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise QueryExecutionError(f"Python code exceeded the {timeout_s:.0f}s time limit.")
    if "err" in outcome:
        exc = outcome["err"]
        raise QueryExecutionError(f"Python error: {type(exc).__name__}: {exc}") from exc  # type: ignore[arg-type]
    if "result" not in env:
        raise QueryExecutionError("Code must assign its final answer to a variable named `result`.")
    return PandasResult(df=_to_frame(env["result"]).head(max_rows), code=code)
