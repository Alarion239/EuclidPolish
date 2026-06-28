"""The typed provenance record model.

Every object the system tracks is one of two things:

* a :class:`Process` — something that *ran* (``GenerationRun``, ``TrainingRun``,
  ``InferenceRun``), carrying the frozen :class:`ConfigSnapshot` that drove it;
* an :class:`Artifact` — something *saved to disk* (a TFRecord, a checkpoint, a
  FITS cutout …), carrying a pointer to the process that produced it.

All records are frozen dataclasses with a ``SCHEMA_VERSION`` and a polymorphic
``to_dict`` / :func:`record_from_dict` JSON round-trip keyed on ``kind``. The
small :class:`Stamp` is the compact subset embedded *inside* a binary artifact
so a bare file is self-identifying even with no sidecar present.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Tuple

from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.ids import ProvId


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Format(str, Enum):
    """The on-disk format of an artifact."""

    TFRECORD = "tfrecord"
    FITS = "fits"
    CSV = "csv"
    CKPT = "ckpt"
    NPY = "npy"


# --------------------------------------------------------------------------- #
# Stamp — the compact thing embedded inside a binary artifact
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Stamp:
    """Self-identifying pointer embedded in an artifact's bytes/header.

    The full :class:`Artifact` record lives in the sidecar; the stamp is the
    minimum needed to recover identity and one-hop lineage from a bare file.
    """

    id: ProvId
    produced_by: Optional[ProvId] = None
    parents: Tuple[ProvId, ...] = ()
    schema_version: int = 3
    subset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "produced_by": str(self.produced_by) if self.produced_by else None,
            "parents": [str(p) for p in self.parents],
            "schema_version": self.schema_version,
            "subset": self.subset,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stamp":
        return cls(
            id=ProvId(d["id"]),
            produced_by=ProvId(d["produced_by"]) if d.get("produced_by") else None,
            parents=tuple(ProvId(p) for p in d.get("parents", [])),
            schema_version=int(d.get("schema_version", 1)),
            subset=d.get("subset"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Stamp":
        return cls.from_dict(json.loads(text))

    @classmethod
    def legacy(cls) -> "Stamp":
        """The stamp for an artifact of unknown (legacy) provenance."""
        return cls(id=ProvId.sentinel(), schema_version=0)


# --------------------------------------------------------------------------- #
# ConfigSnapshot — frozen capture of the config behind a process
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigSnapshot:
    """A frozen snapshot of a typed config dataclass.

    ``config_type`` names the source dataclass (e.g. ``SkySimulatorConfig``)
    and ``fields`` is its field values. The snapshot is documented-by-reference:
    the schema is the named source type, captured verbatim for reproducibility.
    """

    config_type: str
    fields: Dict[str, Any] = field(default_factory=dict)
    SCHEMA_VERSION: ClassVar[int] = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_type": self.config_type,
            "schema_version": self.SCHEMA_VERSION,
            "fields": self.fields,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfigSnapshot":
        return cls(config_type=d["config_type"], fields=dict(d.get("fields", {})))

    @classmethod
    def from_dataclass(cls, obj: Any) -> "ConfigSnapshot":
        return cls(config_type=type(obj).__name__, fields=dataclasses.asdict(obj))


# --------------------------------------------------------------------------- #
# Record base + polymorphic registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, type] = {}


@dataclass(frozen=True)
class ProvRecord:
    """Base of every on-disk provenance object."""

    id: ProvId
    parents: Tuple[ProvId, ...] = ()
    git: Optional[Dict[str, Any]] = field(default=None)
    created_at: str = field(default_factory=_now_iso)

    KIND: ClassVar[str] = "provrecord"
    SCHEMA_VERSION: ClassVar[int] = 3

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Register only classes that declare their own KIND, so the polymorphic
        # loader can dispatch on it.
        if "KIND" in cls.__dict__:
            _REGISTRY[cls.KIND] = cls

    # -- serialization -- #

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "kind": self.KIND,
            "schema_version": self.SCHEMA_VERSION,
            "id": str(self.id),
            "parents": [str(p) for p in self.parents],
            "git": self.git,
            "created_at": self.created_at,
        }
        d.update(self._payload())
        return d

    def _payload(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def _base_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            id=ProvId(d["id"]),
            parents=tuple(ProvId(p) for p in d.get("parents", [])),
            git=d.get("git"),
            created_at=d["created_at"],
        )

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "ProvRecord":
        return cls(**cls._base_kwargs(d))


def record_from_dict(d: Dict[str, Any]) -> ProvRecord:
    """Reconstruct the concrete record for a serialized dict (keyed on ``kind``)."""
    kind = d["kind"]
    try:
        cls = _REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown provenance record kind: {kind!r}") from exc
    return cls._from_dict(d)


# --------------------------------------------------------------------------- #
# Process records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Process(ProvRecord):
    """A thing that ran, with the config that drove it and its I/O lineage."""

    config: Optional[ConfigSnapshot] = None
    inputs: Tuple[ProvId, ...] = ()
    outputs: Tuple[ProvId, ...] = ()
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    status: str = "running"

    KIND: ClassVar[str] = "process"

    def _payload(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict() if self.config else None,
            "inputs": [str(x) for x in self.inputs],
            "outputs": [str(x) for x in self.outputs],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "Process":
        kw = cls._base_kwargs(d)
        kw.update(
            config=ConfigSnapshot.from_dict(d["config"]) if d.get("config") else None,
            inputs=tuple(ProvId(x) for x in d.get("inputs", [])),
            outputs=tuple(ProvId(x) for x in d.get("outputs", [])),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            status=d.get("status", "running"),
        )
        return cls(**kw)


@dataclass(frozen=True)
class GenerationRun(Process):
    KIND: ClassVar[str] = "generationrun"


@dataclass(frozen=True)
class TrainingRun(Process):
    KIND: ClassVar[str] = "trainingrun"


@dataclass(frozen=True)
class InferenceRun(Process):
    KIND: ClassVar[str] = "inferencerun"


# --------------------------------------------------------------------------- #
# Artifact records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Artifact(ProvRecord):
    """A thing saved to disk, pointing back to the process that produced it."""

    produced_by: Optional[ProvId] = None
    format: Optional[Format] = None
    path: Optional[str] = None
    descriptors: Dict[str, Any] = field(default_factory=dict)

    KIND: ClassVar[str] = "artifact"

    def _payload(self) -> Dict[str, Any]:
        return {
            "produced_by": str(self.produced_by) if self.produced_by else None,
            "format": self.format.value if self.format else None,
            "path": self.path,
            "descriptors": self.descriptors,
        }

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        kw = cls._base_kwargs(d)
        kw.update(
            produced_by=ProvId(d["produced_by"]) if d.get("produced_by") else None,
            format=Format(d["format"]) if d.get("format") else None,
            path=d.get("path"),
            descriptors=dict(d.get("descriptors", {})),
        )
        return cls(**kw)


@dataclass(frozen=True)
class SkyTFRecordArtifact(Artifact):
    KIND: ClassVar[str] = "skytfrecordartifact"


@dataclass(frozen=True)
class SourceCatalogArtifact(Artifact):
    KIND: ClassVar[str] = "sourcecatalogartifact"


@dataclass(frozen=True)
class CheckpointArtifact(Artifact):
    KIND: ClassVar[str] = "checkpointartifact"


@dataclass(frozen=True)
class SRCutoutArtifact(Artifact):
    KIND: ClassVar[str] = "srcutoutartifact"
