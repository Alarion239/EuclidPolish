#!/usr/bin/env python
"""Re-run the exact code a tracked backup was made with, in a sandbox.

Time-travel checks out a backup's git commit in a separate worktree and
gives it isolated data/ckpt folders (read-only symlinked inputs, isolated
outputs), then launches that OLD code as a second WebUI — your live work
is never touched. See euclid_polish.tracking.timetravel.

Examples
--------
    # restore a specific model backup (exact code + checkpoint) and open it
    scripts/timetravel.py restore --campaign current --model anchors-on

    # restore an archived campaign at its saved commit, incl. FASRC sandbox
    scripts/timetravel.py restore --campaign anchor-sweep --remote

    scripts/timetravel.py list
    scripts/timetravel.py stop  <short>
    scripts/timetravel.py open  <short>
    scripts/timetravel.py remove <short>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.tracking import TrackingError, default_store, dirty_warning
from euclid_polish.tracking import timetravel as tt
from euclid_polish.web import fasrc_config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _sandbox_cfg(short: str) -> dict:
    cfg = fasrc_config.load()
    base = tt.remote_sandbox_base(cfg.data_dir, short)
    d = cfg.to_dict()
    d.update({
        "repo_path":           tt.remote_worktree_path(cfg.repo_path, short),
        "data_dir":            f"{base}/data",
        "ckpt_dir":            f"{base}/ckpt/wdsr",
        "tracking_remote_dir": f"{base}/tracking",
        "control_socket":      f"/tmp/euclid-polish-tt-{short}.sock",
    })
    return d


def _restore(args) -> int:
    store = default_store()
    try:
        if args.model:
            mm = store.model_backup_meta(args.campaign, args.model) or {}
            commit_info = mm.get("commit")
            seed_dir = store.model_backup_dir(args.campaign, args.model)
        else:
            cm = store.campaign_meta(args.campaign) or {}
            commit_info = cm.get("saved_commit") or cm.get("created_commit")
            seed_dir = None
    except TrackingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not commit_info or not commit_info.get("hash"):
        print("error: no git commit recorded for this backup", file=sys.stderr)
        return 1

    commit = commit_info["hash"]
    try:
        sb = tt.prepare_local_sandbox(
            commit, live_data_dir=Config.DATA_DIR, seed_ckpt_dir=seed_dir,
            source={"campaign": args.campaign, "model": args.model or None})
    except tt.TimeTravelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    short = sb["short"]
    tt.write_home_fasrc_config(short, _sandbox_cfg(short))

    warn = dirty_warning(commit_info)
    if warn:
        print(f"⚠ {warn}", file=sys.stderr)

    if args.remote:
        from euclid_polish.web.remote import SSHConfig, SSHSession
        cfg = fasrc_config.load()
        ssh = SSHSession(SSHConfig(
            user=cfg.ssh_user, host=cfg.ssh_host,
            socket=cfg.control_socket, control_persist=cfg.control_persist))
        push = ["git", "-C", tt.PROJECT_ROOT, "push", "origin",
                f"{commit}:refs/heads/timetravel/{short}"]
        remote = tt.prepare_remote_sandbox(
            ssh, repo_path=cfg.repo_path, data_dir=cfg.data_dir,
            commit=commit, short=short, push_origin_cmd=push)
        tt.set_sandbox_remote(short, remote)
        print("FASRC sandbox:", json.dumps(remote))

    if args.no_open:
        _print(tt.read_sandbox(short))
        return 0
    spawn = tt.spawn_server(short)
    _print(spawn)
    if spawn.get("ok"):
        print(f"\n→ old code running at {spawn['url']}\n")
    return 0 if spawn.get("ok") else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("restore", help="prepare a sandbox + launch old code")
    sp.add_argument("--campaign", default="current",
                    help="'current' or an archived campaign folder name")
    sp.add_argument("--model", default="",
                    help="model backup name (seeds its checkpoint + commit)")
    sp.add_argument("--remote", action="store_true",
                    help="also prepare the FASRC worktree + netscratch sandbox")
    sp.add_argument("--no-open", action="store_true",
                    help="prepare only; don't spawn the second server")

    sub.add_parser("list", help="list sandboxes")
    for c in ("open", "stop", "remove"):
        sp = sub.add_parser(c, help=f"{c} a sandbox")
        sp.add_argument("short")

    args = p.parse_args()
    try:
        if args.cmd == "restore":
            return _restore(args)
        if args.cmd == "list":
            _print(tt.list_sandboxes())
        elif args.cmd == "open":
            _print(tt.spawn_server(args.short))
        elif args.cmd == "stop":
            _print(tt.stop_server(args.short))
        elif args.cmd == "remove":
            _print(tt.remove_sandbox(args.short))
    except tt.TimeTravelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
