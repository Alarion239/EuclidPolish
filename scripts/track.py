#!/usr/bin/env python
"""Command-line front-end to the experiment-tracking store.

Every backup the WebUI can make is also a one-liner here, so results can
be captured from a terminal or a FASRC batch script without the web app
running. The store lives at ``Config.TRACKING_DIR`` (``./tracking`` by
default) and is mirrored to holylabs with ``track.py sync``.

Examples
--------
    # start / end a campaign (the titled folder everything goes into)
    scripts/track.py new --title "anchor sweep" --description "w_anchor=1.0"
    scripts/track.py status
    scripts/track.py save                 # freeze → tracking/archive/<slug>

    # add a note to the markdown notebook
    scripts/track.py log "kicked off 400k-step run, anchors on"

    # back up artifacts (each takes a --comment; --sync also pushes holylabs)
    scripts/track.py model --comment "best val loss" --name anchors-on
    scripts/track.py fits  data/euclid_inference/local_cutout/SR.fits -c "M101 SR"
    scripts/track.py image data/vis/training_log.png -c "loss curve at 120k"

    # mirror the whole store up to persistent holylabs storage
    scripts/track.py sync
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
from euclid_polish.tracking import TrackingError, default_store
from euclid_polish.tracking import sync as tracking_sync


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _do_sync(store) -> int:
    """Build an SSH session from the FASRC config and push → holylabs."""
    from euclid_polish.web import fasrc_config
    from euclid_polish.web.remote import SSHConfig, SSHError, SSHSession

    cfg = fasrc_config.load()
    if not cfg.ssh_user:
        print("ssh_user is unset in ~/.euclid_polish/fasrc.json", file=sys.stderr)
        return 2
    ssh = SSHSession(SSHConfig(
        user=cfg.ssh_user, host=cfg.ssh_host,
        socket=cfg.control_socket, control_persist=cfg.control_persist,
    ))
    if not ssh.is_connected():
        try:
            ssh.connect()
        except SSHError as e:
            print(f"cannot reach FASRC: {e}", file=sys.stderr)
            return 2
    remote_dir = tracking_sync.remote_tracking_dir(
        cfg.repo_path, cfg.tracking_remote_dir,
    )
    res = tracking_sync.push(ssh, remote_dir, local_root=store.root)
    _print_json(res)
    return 0 if res.get("ok") else 1


def _maybe_sync(store, args) -> None:
    if getattr(args, "sync", False):
        _do_sync(store)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("new", help="start a new active campaign")
    sp.add_argument("--title", required=True)
    sp.add_argument("--description", default="")

    sub.add_parser("status", help="show the active campaign + backup counts")
    sub.add_parser("save", help="freeze the active campaign → archive/")
    sub.add_parser("list", help="list active + archived campaigns")
    sub.add_parser("jobs", help="show the active campaign's FASRC job log")
    sub.add_parser("sync", help="rsync the tracking store up to holylabs")

    sp = sub.add_parser("log", help="append a note to the markdown notebook")
    sp.add_argument("text")

    sp = sub.add_parser("model", help="back up a checkpoint dir")
    sp.add_argument("--ckpt-dir", default=Config.DEFAULT_CHECKPOINT_DIR)
    sp.add_argument("-c", "--comment", default="")
    sp.add_argument("--name", default=None)
    sp.add_argument("--sync", action="store_true", help="also push to holylabs")

    for kind in ("fits", "image"):
        sp = sub.add_parser(kind, help=f"back up a {kind} file")
        sp.add_argument("path")
        sp.add_argument("-c", "--comment", default="")
        sp.add_argument("--name", default=None)
        sp.add_argument("--sync", action="store_true",
                        help="also push to holylabs")

    args = p.parse_args()
    store = default_store()

    try:
        if args.cmd == "new":
            _print_json(store.create_campaign(args.title, args.description))
        elif args.cmd == "status":
            cur = store.current()
            if cur is None:
                print("no active campaign — run: track.py new --title ...")
                return 1
            counts = {k: len(v) for k, v in store.list_backups().items()}
            _print_json({"campaign": cur, "backups": counts,
                         "fasrc_jobs": len(store.read_fasrc_jobs())})
        elif args.cmd == "save":
            _print_json(store.save_campaign())
        elif args.cmd == "list":
            _print_json(store.list_campaigns())
        elif args.cmd == "jobs":
            _print_json(store.read_fasrc_jobs())
        elif args.cmd == "log":
            store.append_log(args.text)
            print("logged.")
        elif args.cmd == "model":
            rec = store.backup_model(args.ckpt_dir, args.comment, args.name)
            _print_json(rec)
            _maybe_sync(store, args)
        elif args.cmd == "fits":
            rec = store.backup_fits(args.path, args.comment, args.name)
            _print_json(rec)
            _maybe_sync(store, args)
        elif args.cmd == "image":
            rec = store.backup_image(args.path, args.comment, args.name)
            _print_json(rec)
            _maybe_sync(store, args)
        elif args.cmd == "sync":
            return _do_sync(store)
    except TrackingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
