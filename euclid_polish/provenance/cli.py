"""The ``prov`` command-line interface — walk and query provenance.

    prov show <id>           # the full record
    prov ancestors <id>      # everything upstream (how it was produced)
    prov descendants <id>    # everything downstream (what it touched)
    prov stale --model <id>  # artifacts NOT made by the given current model
    prov rebuild             # rebuild the derived index from sidecars
"""

from __future__ import annotations

import argparse
import json

from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.lineage import Lineage
from euclid_polish.provenance.store import ProvStore


def _kind_of(store: ProvStore, pid: ProvId) -> str:
    rec = store.get_or_none(pid)
    return rec.kind if rec is not None else "?"


def _print_ids(store: ProvStore, ids: list[ProvId]) -> None:
    for pid in sorted(ids, key=str):
        print(f"  {pid}  {_kind_of(store, pid)}")


def main(argv: list[str] | None = None, store: ProvStore | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prov", description="Query provenance lineage.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("show", "ancestors", "descendants"):
        p = sub.add_parser(name)
        p.add_argument("id")
    p_stale = sub.add_parser("stale")
    p_stale.add_argument("--model", required=True, help="the current model's ProvId")
    sub.add_parser("rebuild")

    args = parser.parse_args(argv)
    if store is None:
        store = default_store()
    lineage = Lineage(store)

    if args.cmd == "show":
        rec = store.get_or_none(ProvId(args.id))
        if rec is None:
            print(f"not found: {args.id}")
            return 1
        print(json.dumps(rec.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.cmd == "ancestors":
        anc = lineage.ancestors(ProvId(args.id))
        print(f"{len(anc)} ancestor(s) of {args.id}:")
        _print_ids(store, list(anc))
        return 0

    if args.cmd == "descendants":
        desc = lineage.descendants(ProvId(args.id))
        print(f"{len(desc)} descendant(s) of {args.id}:")
        _print_ids(store, list(desc))
        return 0

    if args.cmd == "stale":
        current = ProvId(args.model)
        stale = [
            rec for rec in store.all_records()
            if lineage.is_stale(rec.id, current) is True
        ]
        print(f"{len(stale)} artifact(s) stale vs current model {current}:")
        for rec in sorted(stale, key=lambda r: str(r.id)):
            print(f"  {rec.id}  {rec.kind}")
        return 0

    if args.cmd == "rebuild":
        store.rebuild_index()
        print(f"index rebuilt: {len(store.all_records())} record(s)")
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
