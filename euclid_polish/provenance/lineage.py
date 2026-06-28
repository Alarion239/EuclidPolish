"""Lineage graph queries over a :class:`~euclid_polish.provenance.store.ProvStore`.

Edges run *upstream* from a record to the things it came from:

* an artifact → its ``produced_by`` process and its ``parents``;
* a process → its ``inputs`` and its ``parents``.

:meth:`Lineage.is_stale` is the headline check — a one-hop comparison of an
artifact's model against the current model.
"""

from __future__ import annotations

from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import ProvRecord
from euclid_polish.provenance.store import ProvStore

# Record kinds that identify "the model" behind an artifact.
_MODEL_KINDS = {"checkpointartifact", "trainingrun"}


def _upstream_ids(rec: ProvRecord) -> list[ProvId]:
    """The ids immediately upstream of ``rec``."""
    ids: list[ProvId] = list(getattr(rec, "parents", ()) or ())
    produced_by = getattr(rec, "produced_by", None)
    if produced_by is not None:
        ids.append(produced_by)
    ids.extend(getattr(rec, "inputs", ()) or ())
    return [i for i in ids if not i.is_sentinel]


class Lineage:
    def __init__(self, store: ProvStore):
        self.store = store

    def ancestors(self, pid: ProvId) -> set[ProvId]:
        """All ids transitively upstream of ``pid`` (excluding ``pid`` itself)."""
        seen: set[ProvId] = set()
        frontier = [pid]
        while frontier:
            current = frontier.pop()
            rec = self.store.get_or_none(current)
            if rec is None:
                continue
            for up in _upstream_ids(rec):
                if up not in seen:
                    seen.add(up)
                    frontier.append(up)
        return seen

    def descendants(self, pid: ProvId) -> set[ProvId]:
        """All ids transitively downstream of ``pid`` (excluding ``pid`` itself)."""
        # Build reverse edges once by scanning the whole store.
        children: dict = {}
        for rec in self.store.all_records():
            for up in _upstream_ids(rec):
                children.setdefault(up, set()).add(rec.id)
        seen: set[ProvId] = set()
        frontier = [pid]
        while frontier:
            current = frontier.pop()
            for child in children.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return seen

    def model_of(self, artifact_id: ProvId) -> ProvId | None:
        """The model id behind ``artifact_id``, or ``None`` if unknown/legacy."""
        rec = self.store.get_or_none(artifact_id)
        if rec is None:
            return None
        candidates = list(getattr(rec, "parents", ()) or ())
        produced_by = getattr(rec, "produced_by", None)
        proc = self.store.get_or_none(produced_by) if produced_by else None
        if proc is not None:
            candidates.extend(getattr(proc, "inputs", ()) or ())
        for cand in candidates:
            if cand.is_sentinel:
                return None
            cand_rec = self.store.get_or_none(cand)
            if cand_rec is not None and cand_rec.kind in _MODEL_KINDS:
                return cand
        return None

    def is_stale(self, artifact_id: ProvId, current_model_id: ProvId) -> bool | None:
        """``True`` if stale, ``False`` if current, ``None`` if unknown.

        Intentionally one-hop: "stale" means *not made by the current model*, not
        "inputs changed".
        """
        model = self.model_of(artifact_id)
        if model is None:
            return None
        return model != current_model_id
