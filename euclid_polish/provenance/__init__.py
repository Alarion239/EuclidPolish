"""Typed provenance & identity for EuclidPolish.

Every generation process (sky generation, training, super-resolution) and
every saved artifact gets an 8-hex :class:`~euclid_polish.provenance.ids.ProvId`
and a frozen, documented record on disk, so any piece of data can be traced
back to exactly how it was produced — and a super-resolved image can be told
apart from a stale one made by an old model.

This package is the lower-level *core*: it has no TensorFlow dependency and
knows nothing about the tracking module (which references it, not the other
way round).
"""

from __future__ import annotations

from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.lineage import Lineage
from euclid_polish.provenance.persistable import Persistable, StampCarrier
from euclid_polish.provenance.records import (
    Artifact,
    CheckpointArtifact,
    ConfigSnapshot,
    Format,
    GenerationRun,
    InferenceRun,
    Process,
    ProvRecord,
    SRCutoutArtifact,
    SkyTFRecordArtifact,
    SourceCatalogArtifact,
    Stamp,
    TrainingRun,
    record_from_dict,
)
from euclid_polish.provenance.store import ProvStore

__all__ = [
    "ProvId",
    "ProvStore",
    "Lineage",
    "Persistable",
    "StampCarrier",
    "Stamp",
    "ConfigSnapshot",
    "Format",
    "ProvRecord",
    "Process",
    "Artifact",
    "GenerationRun",
    "TrainingRun",
    "InferenceRun",
    "SkyTFRecordArtifact",
    "SourceCatalogArtifact",
    "CheckpointArtifact",
    "SRCutoutArtifact",
    "record_from_dict",
    "capture_git",
]
