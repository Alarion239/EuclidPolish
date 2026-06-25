"""The :class:`ProvId` value type — an 8-hex provenance identifier.

Eight hex characters give 16**8 ≈ 4.3 billion ids; because the store keeps one
sidecar per object, ids are *minted-then-checked* (re-drawn on the rare clash)
so collisions are impossible, not merely unlikely. The id is short enough to
sit in a filename for greppability (``clean_train.4b1e7a90.tfrecord``).
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Callable

# Eight lowercase hex characters.
_ID_RE = re.compile(r"[0-9a-f]{8}")

# Reserved id meaning "provenance unknown" (legacy / un-stamped artifacts).
_SENTINEL = "00000000"


@dataclass(frozen=True)
class ProvId:
    """An 8-hex provenance identifier.

    Construction validates and normalises to lowercase, so ``ProvId("4B1E7A90")``
    and ``ProvId("4b1e7a90")`` compare equal and hash identically.
    """

    value: str

    def __post_init__(self) -> None:
        normalised = self.value.lower()
        if not _ID_RE.fullmatch(normalised):
            raise ValueError(
                f"ProvId must be 8 hex characters, got {self.value!r}"
            )
        # Frozen dataclass: bypass the immutability guard to store the
        # normalised form.
        object.__setattr__(self, "value", normalised)

    def __str__(self) -> str:
        return self.value

    def as_filename_token(self) -> str:
        """The token to embed in an artifact's filename."""
        return self.value

    @property
    def is_sentinel(self) -> bool:
        """``True`` for the reserved "unknown provenance" id."""
        return self.value == _SENTINEL

    @classmethod
    def sentinel(cls) -> "ProvId":
        """The reserved id meaning provenance is unknown (legacy artifacts)."""
        return cls(_SENTINEL)

    @classmethod
    def parse(cls, text: str) -> "ProvId":
        """Parse a string into a :class:`ProvId` (alias for the constructor)."""
        return cls(text)

    @classmethod
    def mint(cls, exists: Callable[["ProvId"], bool]) -> "ProvId":
        """Draw a fresh id, re-drawing while ``exists(id)`` is true.

        ``exists`` is the collision predicate — typically ``ProvStore.exists`` —
        so a freshly minted id is guaranteed absent from the store. The sentinel
        id is never minted.
        """
        while True:
            candidate = cls(secrets.token_hex(4))
            if candidate.is_sentinel:
                continue
            if not exists(candidate):
                return candidate
