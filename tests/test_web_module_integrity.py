"""Static integrity guard for the split web layer (routes/ + helpers/).

The app.py refactor moved routes into per-group ``register(app)`` modules
and helpers into ``helpers/``. A name used in one route module but only
*defined* in another (e.g. a nested helper that turned out to be shared)
imports fine and registers fine — it only blows up with ``NameError`` when
that specific route body runs. Most route bodies are SSH-gated, so the
HTTP-level tests never execute them and such bugs slip through.

This test parses every route/helper module and asserts there are no
undefined free names — catching cross-module references and missing
imports at collection time, before they reach production.
"""

from __future__ import annotations

import ast
import builtins
import glob
import os

import euclid_polish.web.app as web_app

_WEB_DIR = os.path.dirname(os.path.abspath(web_app.__file__))
_ALLOWED = set(dir(builtins)) | {
    "__name__", "__file__", "__doc__",
    "app",                 # the register(app) parameter
    "self", "cls",
}


def _undefined_names(path: str) -> list[str]:
    tree = ast.parse(open(path).read())
    defined = set(_ALLOWED)
    used: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                defined.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                defined.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, ast.Name):
            (used if isinstance(n.ctx, ast.Load) else defined).add(n.id)
    return sorted(used - defined)


def test_no_undefined_names_in_web_modules():
    offenders = {}
    for sub in ("routes", "helpers"):
        for path in glob.glob(os.path.join(_WEB_DIR, sub, "*.py")):
            undef = _undefined_names(path)
            if undef:
                offenders[os.path.relpath(path, _WEB_DIR)] = undef
    assert not offenders, (
        "Undefined free names in web modules (cross-module reference or "
        f"missing import): {offenders}"
    )
