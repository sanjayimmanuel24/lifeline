"""run_code tool: execute LLM-written analysis code in isolation.

Docker deployment: a separate `sandbox` container on an internal network with no internet,
read-only filesystem, 1 CPU / 512 MB, reached over HTTP (SANDBOX_URL).
Local fallback: a separate Python process (-I isolated mode) in a temp dir with a timeout.
Both paths first reject code that imports anything outside a whitelist (AST check).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "json", "datetime", "collections"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "breakpoint", "exit", "quit"}
TIMEOUT_S = 10

RUNNER = r'''
import json, sys, io, contextlib
import pandas as pd, numpy as np
records = pd.read_json(sys.argv[1], orient="records")
calls = pd.read_json(sys.argv[2], orient="records")
zones = json.load(open(sys.argv[3]))
code = open(sys.argv[4]).read()
buf = io.StringIO()
env = {"pd": pd, "np": np, "records": records, "calls": calls, "zones": zones}
with contextlib.redirect_stdout(buf):
    exec(compile(code, "<agent>", "exec"), env)
res = env.get("result")
if isinstance(res, (pd.DataFrame, pd.Series)):
    res = res.reset_index().to_dict(orient="records") if isinstance(res, pd.DataFrame) else res.to_dict()
print("@@RESULT@@" + json.dumps({"stdout": buf.getvalue()[-2000:], "result": res}, default=str))
'''


def check(code: str) -> str | None:
    """Static safety check. Returns an error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            bad = [m for m in mods if m not in ALLOWED_IMPORTS]
            if bad:
                return f"import of {bad} is not allowed"
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not allowed"
        if isinstance(node, ast.Attribute) and node.attr in {"to_csv", "to_pickle", "to_parquet", "read_csv", "system"}:
            return f"file/system access '{node.attr}' is not allowed"
    return None


def run_code(code: str, records: list[dict], calls: list[dict], zones: list[str]) -> dict:
    err = check(code)
    if err:
        return {"ok": False, "error": err, "where": "static-check"}
    url = os.getenv("SANDBOX_URL")
    if url:
        try:
            r = httpx.post(f"{url}/run", json={"code": code, "records": records, "calls": calls, "zones": zones},
                           timeout=TIMEOUT_S + 5)
            return r.json() | {"where": "sandbox-container"}
        except httpx.HTTPError as e:
            return {"ok": False, "error": f"sandbox unreachable: {e}", "where": "sandbox-container"}
    return _run_local(code, records, calls, zones)


def _run_local(code: str, records, calls, zones) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "records.json").write_text(json.dumps(records, default=str))
        (p / "calls.json").write_text(json.dumps(calls, default=str))
        (p / "zones.json").write_text(json.dumps(zones))
        (p / "code.py").write_text(code)
        (p / "runner.py").write_text(RUNNER)
        try:
            proc = subprocess.run([sys.executable, "-I", "runner.py", "records.json", "calls.json", "zones.json", "code.py"],
                                  cwd=d, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {TIMEOUT_S}s", "where": "subprocess"}
        out = proc.stdout
        if "@@RESULT@@" not in out:
            return {"ok": False, "error": (proc.stderr or out)[-800:], "where": "subprocess"}
        payload = json.loads(out.split("@@RESULT@@", 1)[1])
        return {"ok": True, **payload, "where": "subprocess"}
