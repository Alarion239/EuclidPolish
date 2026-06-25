"""Provenance helpers for the synthetic generation pipeline.

A generation run (one ``run_pipeline`` invocation, per subset) mints a
:class:`GenerationRun` whose config is frozen from the live
``MultiBandGeneratorConfig`` / ``MultiBandForwardConfig``. Every record it
writes is stamped with its *file-level* artifact id (so all records in
``clean_train.tfrecord`` share one id, addressed individually by their index)
plus the run id. The whole thing is best-effort: :func:`make_generation_context`
returns ``None`` if anything goes wrong, and callers write unstamped records
exactly as before.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import (
    ConfigSnapshot, Format, GenerationRun, SkyTFRecordArtifact,
)
from euclid_polish.provenance.store import ProvStore
from euclid_polish.provenance.records import Stamp


@dataclass
class GenerationContext:
    """Shared identity for one generation run and the files it writes."""

    store: ProvStore
    run_id: ProvId
    git: Optional[Dict[str, Any]] = None
    _file_ids: Dict[Tuple[str, str], ProvId] = field(default_factory=dict)

    def file_id(self, kind: str, subset: str) -> ProvId:
        """The (stable) artifact id for the ``kind``/``subset`` record file."""
        key = (kind, subset)
        fid = self._file_ids.get(key)
        if fid is None:
            fid = self.store.mint()
            self._file_ids[key] = fid
        return fid

    def stamp(self, kind: str, subset: str, *,
              parents: Sequence[ProvId] = ()) -> Stamp:
        """The stamp every record of this ``kind``/``subset`` carries."""
        return Stamp(
            id=self.file_id(kind, subset),
            produced_by=self.run_id,
            parents=tuple(parents),
            schema_version=3,
            subset=subset,
        )

    def finalize(self, kind: str, subset: str, path: str, *,
                 parents: Sequence[ProvId] = ()) -> SkyTFRecordArtifact:
        """Persist the file-level :class:`SkyTFRecordArtifact` next to the data."""
        art = SkyTFRecordArtifact(
            id=self.file_id(kind, subset),
            git=self.git,
            produced_by=self.run_id,
            format=Format.TFRECORD,
            path=path,
            parents=tuple(parents),
            descriptors={"kind": kind, "subset": subset},
        )
        self.store.put(art, sidecar_dir=os.path.dirname(path) or None)
        return art


@dataclass(frozen=True)
class ShardStampPlan:
    """Pre-minted ids for one subset of a parallel generation run.

    Picklable and store-free, so the parent mints the ids once and ships the
    plan to each worker process; workers stamp records without touching a store.
    The three record files of a subset share one id each; hr/dirty parent on the
    clean file.
    """

    run_id: ProvId
    clean_id: ProvId
    hr_id: ProvId
    dirty_id: ProvId

    def clean_stamp(self, subset: str) -> Stamp:
        return Stamp(id=self.clean_id, produced_by=self.run_id,
                     schema_version=3, subset=subset)

    def hr_stamp(self, subset: str) -> Stamp:
        return Stamp(id=self.hr_id, produced_by=self.run_id,
                     parents=(self.clean_id,), schema_version=3, subset=subset)

    def dirty_stamp(self, subset: str) -> Stamp:
        return Stamp(id=self.dirty_id, produced_by=self.run_id,
                     parents=(self.clean_id,), schema_version=3, subset=subset)


def begin_generation_run(store: ProvStore, cfg: Any, *,
                         parents: Sequence[ProvId] = (),
                         git: Optional[Dict[str, Any]] = None) -> GenerationContext:
    """Mint + persist a :class:`GenerationRun` and return its context."""
    run = GenerationRun(
        id=store.mint(),
        git=git,
        parents=tuple(parents),
        config=ConfigSnapshot.from_dataclass(cfg),
        status="ok",
    )
    store.put(run)
    return GenerationContext(store=store, run_id=run.id, git=git)


def make_generation_context(cfg: Any, *,
                            parents: Sequence[ProvId] = ()) -> Optional[GenerationContext]:
    """Best-effort context using the project's default store; ``None`` on failure."""
    try:
        store = default_store()
        return begin_generation_run(store, cfg, parents=parents, git=capture_git())
    except Exception:
        return None
