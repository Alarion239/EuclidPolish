"""The FASRC gate is per route (contract C4), not a prefix allowlist.

Handlers that need the SSH session carry ``@requires_fasrc``; with FASRC
disconnected they answer ``503 {"ok": false, "error": "FASRC not connected",
"code": "fasrc_offline"}`` and every other endpoint keeps working offline.
"""

from __future__ import annotations

import ast
import functools
import os
import signal
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from flask import Flask, jsonify

from euclid_polish.config import Config
from euclid_polish.web import app as web_app
from euclid_polish.web import fasrc_gate, remote
from euclid_polish.web.fasrc_gate import (
    FASRC_OFFLINE_PAYLOAD,
    fasrc_connected,
    register_fasrc_gate,
    requires_fasrc,
)
from euclid_polish.web.helpers.paths import _safe_relpath

WEB = Path(web_app.__file__).parent
PACKAGE = WEB.parent
REPO = PACKAGE.parent
ROUTE_SOURCES = sorted((WEB / "routes").glob("*.py")) + [WEB / "app.py"]

OFFLINE = {"ok": False, "error": "FASRC not connected", "code": "fasrc_offline"}


class _Up:
    def is_connected(self) -> bool:
        return True

    def run(self, cmd, timeout=None, binary=False):
        return (0, "", "")


class _Down:
    def is_connected(self) -> bool:
        return False

    def run(self, *_a, **_kw):
        raise AssertionError("a gated handler must not run while disconnected")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    monkeypatch.setattr(remote.STATE, "connected_at", None)


@pytest.fixture
def app(offline):
    application = web_app.create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# The decorator and the hook on a bare app
# ---------------------------------------------------------------------------

def _bare_app() -> Flask:
    bare = Flask(__name__)
    register_fasrc_gate(bare)

    @bare.get("/remote")
    @requires_fasrc
    def remote_view():
        return jsonify({"ok": True})

    @bare.post("/remote-post")
    @requires_fasrc
    def remote_post():
        return jsonify({"ok": True})

    @bare.get("/local")
    def local_view():
        return jsonify({"ok": True, "local": True})

    return bare


def test_requires_fasrc_marks_and_returns_the_same_function():
    def view():
        return "x"

    assert requires_fasrc(view) is view
    assert view._requires_fasrc is True


def test_offline_payload_is_the_contract():
    assert FASRC_OFFLINE_PAYLOAD == OFFLINE


def test_marked_view_is_refused_while_disconnected(offline):
    c = _bare_app().test_client()
    for response in (c.get("/remote"), c.post("/remote-post")):
        assert response.status_code == 503
        assert response.get_json() == OFFLINE


def test_marked_view_runs_while_connected(monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    assert fasrc_connected() is True
    assert _bare_app().test_client().get("/remote").get_json() == {"ok": True}


def test_dropped_session_counts_as_disconnected(monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Down())
    assert fasrc_connected() is False
    assert _bare_app().test_client().get("/remote").status_code == 503


def test_unmarked_and_unknown_paths_ignore_the_gate(offline):
    c = _bare_app().test_client()
    assert c.get("/local").get_json() == {"ok": True, "local": True}
    assert c.get("/nope").status_code == 404
    assert c.post("/local").status_code == 405


def test_gate_decorator_order_does_not_matter(offline):
    bare = Flask(__name__)
    register_fasrc_gate(bare)

    @requires_fasrc
    @bare.get("/outer")
    def outer():
        return "ran"

    assert bare.test_client().get("/outer").status_code == 503


# ---------------------------------------------------------------------------
# The real app with FASRC disconnected
# ---------------------------------------------------------------------------

def _write_fits(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.arange(64, dtype=np.float32).reshape(8, 8)).writeto(path)


def test_old_prefix_gate_is_gone():
    source = (WEB / "app.py").read_text()
    assert "_enforce_ssh_gate" not in source
    assert "_ALWAYS_REACHABLE_PREFIXES" not in source


@pytest.mark.parametrize("path", [
    "/api/git/status",
    "/api/jobs",
    "/api/jobs?summary=1",
    "/api/version",
    "/api/fasrc/status",
    "/api/fasrc/config",
    "/api/noise",
    "/api/fasrc/steps/status",
    "/api/tracking/state",
    "/api/cutouts/VIS/list.json",
    "/ensemble/status.json",
])
def test_local_endpoints_work_offline(client, path):
    response = client.get(path)
    assert response.status_code == 200, response.get_data(as_text=True)[:300]


def test_inspector_preview_and_download_work_offline(client):
    fits_path = Path(Config.VIS_DIR) / "gate" / "probe.fits"
    _write_fits(fits_path)
    rel = _safe_relpath(os.path.realpath(fits_path))

    preview = client.get(f"/inspect/preview.png?fits={rel}&size=32")
    assert preview.status_code == 200
    assert preview.data[:8] == b"\x89PNG\r\n\x1a\n"
    download = client.get(f"/inspect/download?fits={rel}")
    assert download.status_code == 200
    assert download.data[:6] == b"SIMPLE"


def test_figure_images_work_offline(client):
    png = Path(Config.VIS_DIR) / "gate" / "probe.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    response = client.get("/vis/gate/probe.png")
    assert response.status_code == 200
    assert response.data.startswith(b"\x89PNG")


def test_connection_retry_reports_its_own_error_offline(client):
    response = client.post("/api/connection/retry")
    body = response.get_json()
    assert response.status_code != 503
    assert body["ok"] is False
    assert body.get("code") != "fasrc_offline"
    assert "auto-connect disabled" in body["error"]


def test_offline_non_api_pages_are_not_redirected_to_a_connection_page(client):
    response = client.get("/cutout-image/NOPE/star_0000_512.fits")
    assert response.status_code == 404


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/fasrc/steps/euclid_query/submit"),
    ("GET", "/api/fasrc/queue"),
    ("GET", "/api/fasrc/current-submission"),
    ("GET", "/api/fasrc/runs"),
    ("GET", "/api/fasrc/git-status"),
    ("POST", "/api/fasrc/cancel"),
    ("POST", "/api/fasrc/env-update"),
    ("POST", "/api/sky/sync"),
    ("POST", "/api/evaluation/sync"),
    ("POST", "/ensemble/pull"),
    ("POST", "/tng-auth/save"),
    ("POST", "/euclid-auth/save"),
    ("POST", "/api/euclid-psf/sync"),
    ("POST", "/api/tracking/sync"),
    ("GET", "/fasrc/file/download?remote_path=/n/x"),
    ("GET", "/poster/result/cutout.png"),
    ("GET", "/poster/result/cutout.fits"),
    ("GET", "/tng/result/grid.png"),
])
def test_marked_endpoints_return_the_offline_payload(client, method, path):
    response = client.open(path, method=method)
    assert response.status_code == 503
    assert response.get_json() == OFFLINE


# ---------------------------------------------------------------------------
# Audit: every SSH-touching handler is marked or knowingly degrades
# ---------------------------------------------------------------------------

# Handlers that reference the SSH session but work (degrade) offline. Each
# entry says why it must NOT be gated.
GRACEFUL = {
    "api_fasrc_connect": "the connect action itself",
    "api_fasrc_disconnect": "closing an already-closed session is a no-op",
    "api_fasrc_steps_status": "serves the static step schema offline",
    "api_fasrc_job_status": "returns an empty status offline",
    "euclid_auth_status": "reports connected=false offline",
    "tng_auth_status": "reports connected=false offline",
    "tng_radii_status": "read-only: answers from the cache (connected flag)",
    "api_timetravel_restore": "local sandbox; the remote half is optional",
    "api_population_comparison_sync_training_catalog":
        "its job self-connects (ensure_ssh_connected) and reports failure",
    "api_archive_fields_sync":
        "its job self-connects (ensure_ssh_connected) and reports failure",
    "tng_histograms_png": "renders the local cache; FASRC ids/key are optional",
    "api_tracking_state": "local store; reports ssh_connected=false offline",
    "api_tracking_save": "local save; the holylabs push is best-effort",
    "api_tracking_backup": "local backup; the holylabs push is best-effort",
    "api_connection_retry": "the connect action itself (C4: works offline)",
    "ensemble_archive_member": ("local archive job; the FASRC copy's delete "
                                "is best-effort and skipped offline"),
}

_SSH_METHODS = {"run", "stream", "rsync_pull", "rsync_push", "write_text"}
# Method names common outside SSHSession (``Path.write_text``): counted only
# when the receiver is visibly an SSH session (``ssh.write_text(...)``).
_SSH_ONLY_METHODS = {"write_text"}
_SSH_NAMES = {
    "fetch_one_file", "ensure_ssh_connected", "list_remote_dir",
    "run_remote_python", "rsync_pull", "rsync_push",
}


def _is_route(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr in {"route", "get", "post"}
    )


def _is_marked(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "requires_fasrc":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "requires_fasrc":
            return True
    return False


def _body(node: ast.FunctionDef) -> list[ast.stmt]:
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                      # ignore the docstring
    return body


def _direct_ssh_hits(node: ast.FunctionDef) -> set[str]:
    hits = set()
    for stmt in _body(node):
        for sub in ast.walk(stmt):
            if (isinstance(sub, ast.Attribute) and sub.attr == "ssh"
                    and isinstance(sub.value, ast.Name) and sub.value.id == "STATE"):
                hits.add("STATE.ssh")
            elif (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                  and sub.func.attr in _SSH_METHODS
                  and not (isinstance(sub.func.value, ast.Name)
                           and sub.func.value.id == "subprocess")
                  and (sub.func.attr not in _SSH_ONLY_METHODS
                       or "ssh" in ast.unparse(sub.func.value).lower())):
                hits.add(f".{sub.func.attr}(")
            elif isinstance(sub, ast.Name) and sub.id in _SSH_NAMES:
                hits.add(sub.id)
            elif isinstance(sub, ast.Attribute) and sub.attr in _SSH_NAMES:
                hits.add(sub.attr)
    return hits


def _local_defs(node: ast.FunctionDef) -> set[str]:
    """Names of the functions nested inside ``node`` (their bodies are
    already part of ``node``'s own walk)."""
    return {
        sub.name for sub in ast.walk(node)
        if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef) and sub is not node
    }


@dataclass
class _Module:
    """One parsed module: its functions (every def, by name) and imports."""

    name: str
    tree: ast.Module
    functions: dict[str, list[ast.FunctionDef]]
    imported: dict[str, tuple[str, str]]   # local name → (module, attribute)
    aliases: dict[str, str]                # local name → module

    @classmethod
    def parse(cls, name: str, source: str, *, is_package: bool,
              known: set[str]) -> _Module:
        tree = ast.parse(source)
        functions: dict[str, list[ast.FunctionDef]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.setdefault(node.name, []).append(node)
        package = name if is_package else name.rpartition(".")[0]
        imported: dict[str, tuple[str, str]] = {}
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package.split(".")[:len(package.split(".")) - node.level + 1]
                    base = ".".join([*anchor, base] if base else anchor)
                for alias in node.names:
                    local = alias.asname or alias.name
                    if f"{base}.{alias.name}" in known:
                        aliases[local] = f"{base}.{alias.name}"
                    elif base in known:
                        imported[local] = (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in known and alias.asname:
                        aliases[alias.asname] = alias.name
        return cls(name, tree, functions, imported, aliases)

    def references(self, node: ast.FunctionDef) -> set[tuple[str, str]]:
        """``(module, name)`` of every function ``node`` calls *or* passes
        on (a job target handed to ``REGISTRY.spawn`` runs later)."""
        local = _local_defs(node)
        refs: set[tuple[str, str]] = set()
        for stmt in _body(node):
            for sub in ast.walk(stmt):
                if (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
                        and sub.id not in local):
                    if sub.id in self.imported:
                        refs.add(self.imported[sub.id])
                    elif sub.id in self.functions:
                        refs.add((self.name, sub.id))
                elif (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                      and sub.value.id in self.aliases):
                    refs.add((self.aliases[sub.value.id], sub.attr))
        return refs


def _short(key: tuple[str, str]) -> str:
    return f"{key[0].rpartition('.')[2]}.{key[1]}"


class _SshIndex:
    """Which functions reach SSH, directly or through any chain of helpers
    in *any* module (``status._catalog_status → fetch_one_file``), so a
    handler cannot hide its SSH use behind a cross-module helper."""

    def __init__(self, sources: dict[str, tuple[str, bool]]) -> None:
        known = set(sources)
        self.modules = {
            name: _Module.parse(name, source, is_package=is_package, known=known)
            for name, (source, is_package) in sources.items()
        }
        refs: dict[tuple[str, str], set[tuple[str, str]]] = {}
        self.reach: dict[tuple[str, str], str] = {}
        for module in self.modules.values():
            for name, defs in module.functions.items():
                key = (module.name, name)
                direct = set().union(*(_direct_ssh_hits(d) for d in defs))
                if direct:
                    self.reach[key] = sorted(direct)[0]
                refs[key] = set().union(*(module.references(d) for d in defs))
        changed = True
        while changed:                       # fixed point over the call graph
            changed = False
            for key, targets in refs.items():
                if key in self.reach:
                    continue
                for target in sorted(self.resolve(t) for t in targets):
                    if target in self.reach:
                        self.reach[key] = f"{_short(target)}→{self.reach[target]}"
                        changed = True
                        break

    def resolve(self, key: tuple[str, str], depth: int = 0) -> tuple[str, str]:
        """Follow re-exports (``from .status import x`` in a package)."""
        module = self.modules.get(key[0])
        if (module is None or key[1] in module.functions
                or key[1] not in module.imported or depth > 8):
            return key
        return self.resolve(module.imported[key[1]], depth + 1)

    def handler_hits(self, module_name: str, node: ast.FunctionDef) -> list[str]:
        module = self.modules[module_name]
        hits = set(_direct_ssh_hits(node))
        for target in module.references(node):
            target = self.resolve(target)
            if target in self.reach:
                hits.add(f"{_short(target)}→{self.reach[target]}")
        return sorted(hits)


def _package_sources() -> dict[str, tuple[str, bool]]:
    """``{dotted module: (source, is_package)}`` for all of ``euclid_polish``
    except the frontend tree."""
    sources = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(REPO).with_suffix("")
        if "frontend" in rel.parts or "node_modules" in rel.parts:
            continue
        is_package = rel.name == "__init__"
        parts = rel.parts[:-1] if is_package else rel.parts
        sources[".".join(parts)] = (path.read_text(encoding="utf-8"), is_package)
    return sources


@functools.cache
def _package_index() -> _SshIndex:
    return _SshIndex(_package_sources())


def _module_name(source: Path) -> str:
    parts = source.relative_to(REPO).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _route_handlers():
    for source in ROUTE_SOURCES:
        module = _package_index().modules[_module_name(source)]
        for node in ast.walk(module.tree):
            if (isinstance(node, ast.FunctionDef)
                    and any(_is_route(d) for d in node.decorator_list)):
                yield source.name, module.name, node


def test_every_ssh_touching_handler_is_marked_or_graceful():
    unaccounted = []
    for filename, module_name, node in _route_handlers():
        hits = _package_index().handler_hits(module_name, node)
        if hits and not _is_marked(node) and node.name not in GRACEFUL:
            unaccounted.append(f"{filename}:{node.lineno} {node.name} {hits}")
    assert unaccounted == []


def _index(modules: dict[str, str], packages: tuple[str, ...] = ()) -> _SshIndex:
    return _SshIndex({
        name: (textwrap.dedent(source), name in packages)
        for name, source in modules.items()
    })


def _handler(index: _SshIndex, module: str, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(index.modules[module].tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_audit_follows_same_module_helpers():
    index = _index({"pkg.routes": """
        def register(app):
            def _pull(name):
                return fetch_one_file(name, force=True)
            def _serve(name):
                return _pull(name)
            @app.route('/x')
            def x():
                return _serve('a')
    """})
    hits = index.handler_hits("pkg.routes", _handler(index, "pkg.routes", "x"))
    assert hits == ["routes._serve→routes._pull→fetch_one_file"]


def test_audit_follows_cross_module_helpers_and_job_targets():
    index = _index({
        "pkg": "from pkg.status import pull\n",
        "pkg.status": """
            from pkg.remote import STATE
            def _dir(force=True):
                return fetch_one_file("x", force=force)
            def pull():
                return _dir()
            def rsync_job(cap):
                return STATE.ssh
            def local():
                return 1
        """,
        "pkg.routes": """
            from pkg import status as helpers
            from pkg import pull as reexported
            from pkg.status import local, pull, rsync_job
            def register(app):
                @app.route('/by-name')
                def by_name():
                    return pull()
                @app.route('/by-alias')
                def by_alias():
                    return helpers.pull()
                @app.route('/job')
                def job():
                    return REGISTRY.spawn("x", rsync_job)
                @app.route('/reexport')
                def reexport():
                    return reexported()
                @app.route('/local')
                def local_only():
                    return local()
        """,
    }, packages=("pkg",))

    def hits(name):
        return index.handler_hits("pkg.routes", _handler(index, "pkg.routes", name))

    assert hits("by_name") == ["status.pull→status._dir→fetch_one_file"]
    assert hits("by_alias") == ["status.pull→status._dir→fetch_one_file"]
    assert hits("job") == ["status.rsync_job→STATE.ssh"]
    assert hits("reexport") == ["status.pull→status._dir→fetch_one_file"]
    assert hits("local_only") == []


def test_audit_sees_the_real_cross_module_helpers():
    """The helpers the old same-module audit could not see are indexed."""
    reach = _package_index().reach
    for key in [
        ("euclid_polish.web.helpers.status", "_catalog_status"),
        ("euclid_polish.web.helpers.status", "_fasrc_catalog_dir"),
        ("euclid_polish.web.helpers.status", "_valid_4band_stars"),
        ("euclid_polish.web.helpers.status", "_ensure_local_star_cutout"),
        ("euclid_polish.web.helpers.status", "_fasrc_psf_dir"),
        ("euclid_polish.web.helpers.ensemble_viz", "job_ensemble_pull"),
        ("euclid_polish.web.fasrc_jobs", "submit_sbatch_script"),
        ("euclid_polish.tracking.sync", "push"),
        ("euclid_polish.tracking.timetravel", "prepare_remote_sandbox"),
    ]:
        assert key in reach, key


# ---------------------------------------------------------------------------
# Runtime twin of the audit: sweep every argument-free GET offline
# ---------------------------------------------------------------------------

_SSH_PROGRAMS = {"ssh", "rsync", "scp", "sftp"}
_SWEEP_LIMIT_S = 20


class _SweepTimeout(Exception):
    pass


def _ssh_argv(args, kwargs) -> str | None:
    """The command line when a subprocess call would start an SSH program."""
    argv = kwargs.get("args", args[0] if args else None)
    if isinstance(argv, str):
        argv = argv.split()
    if (isinstance(argv, list | tuple) and argv
            and os.path.basename(str(argv[0])) in _SSH_PROGRAMS):
        return " ".join(map(str, argv))[:100]
    return None


@pytest.fixture
def ssh_blocked(monkeypatch):
    """Record (and refuse) every ssh/rsync/scp/sftp subprocess."""
    attempts: list[str] = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def _check(args, kwargs) -> None:
        command = _ssh_argv(args, kwargs)
        if command is not None:
            attempts.append(command)
            raise OSError(f"SSH blocked by the offline sweep: {command}")

    def run(*args, **kwargs):
        _check(args, kwargs)
        return real_run(*args, **kwargs)

    class Popen(real_popen):
        def __init__(self, *args, **kwargs):
            _check(args, kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", Popen)
    return attempts


def _on_alarm(_signum, _frame):
    raise _SweepTimeout()


def test_every_argument_free_get_answers_offline_without_ssh(app, ssh_blocked):
    """With FASRC disconnected and every SSH subprocess blocked, each GET
    route without URL arguments is requested once: a marked route must give
    the offline 503, an unmarked one must never try SSH (not even a
    self-connect) and must answer within the limit. Routes with URL
    arguments and non-GET methods are covered by the AST audit only.
    (Handler errors unrelated to SSH, e.g. a write the test data-guard
    refuses, are not this test's concern.)"""
    client = app.test_client()
    problems = []
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            if ("GET" not in (rule.methods or ()) or rule.arguments
                    or rule.endpoint == "static"):
                continue
            gated = getattr(app.view_functions[rule.endpoint], "_requires_fasrc", False)
            ssh_blocked.clear()
            status, payload = None, None
            signal.alarm(_SWEEP_LIMIT_S)
            try:
                response = client.get(rule.rule)
                status, payload = response.status_code, response.get_json(silent=True)
            except _SweepTimeout:
                problems.append(f"{rule.rule}: no answer within {_SWEEP_LIMIT_S} s")
            except Exception:  # noqa: BLE001 - an unrelated handler error
                status = "error"
            finally:
                signal.alarm(0)
            if gated and (status, payload) != (503, OFFLINE):
                problems.append(f"{rule.rule}: marked but answered {status}")
            if not gated and ssh_blocked:
                problems.append(f"{rule.rule}: unmarked but tried {ssh_blocked[:2]}")
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert problems == []


def test_sweep_blocks_and_records_ssh_programs(ssh_blocked):
    with pytest.raises(OSError, match="blocked"):
        subprocess.run(["ssh", "-O", "check", "host"])
    with pytest.raises(OSError, match="blocked"):
        subprocess.Popen(["/usr/bin/rsync", "-a", "x", "y"])
    assert subprocess.run(["true"]).returncode == 0
    assert ssh_blocked == ["ssh -O check host", "/usr/bin/rsync -a x y"]


def test_graceful_allowlist_has_no_stale_or_marked_entries():
    handlers = {node.name: node for _file, _module, node in _route_handlers()}
    for name in GRACEFUL:
        assert name in handlers, f"{name} no longer exists"
        assert not _is_marked(handlers[name]), f"{name} is both marked and graceful"


def test_marked_handlers_carry_the_runtime_flag(app):
    marked = {node.name for _file, _module, node in _route_handlers() if _is_marked(node)}
    assert marked, "no handler is marked"
    flagged = {
        getattr(view, "__name__", endpoint)
        for endpoint, view in app.view_functions.items()
        if getattr(view, "_requires_fasrc", False)
    }
    assert marked == flagged


def test_gate_module_exports_the_contract():
    assert set(fasrc_gate.__all__) >= {
        "FASRC_OFFLINE_PAYLOAD", "fasrc_connected", "register_fasrc_gate",
        "requires_fasrc",
    }
