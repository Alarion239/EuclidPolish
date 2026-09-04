"""Validated, immutable amplitude calibration for delivered-MER VIS noise.

The calibration rescales only a stochastic detector-noise residual.  It does
not spatially filter that residual, so the native white-noise structure and
the relative signal-dependent Poisson variance are retained.
``residual_scale`` is the calibrated median absolute robust RMS in electrons
per MER pixel, and ``field_scale_quantiles`` restores the measured
field-to-field spread. Runtime loading is deliberately strict, versioned, and
fingerprinted so generated data cannot silently use the retired correlated
noise schema or a partially edited calibration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


def _validated_field_scale_quantiles(
    value: Sequence[float],
) -> tuple[float, float, float, float, float]:
    try:
        quantiles = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("field_scale_quantiles must contain five numbers") from exc
    if quantiles.ndim != 1 or quantiles.shape != (5,):
        raise ValueError("field_scale_quantiles must contain exactly five values")
    if not np.all(np.isfinite(quantiles)) or np.any(quantiles <= 0.0):
        raise ValueError("field_scale_quantiles must be finite and positive")
    if np.any(np.diff(quantiles) < 0.0):
        raise ValueError("field_scale_quantiles must be nondecreasing")
    if not np.isclose(quantiles[2], 1.0, rtol=0.0, atol=0.02):
        raise ValueError(
            "median field_scale_quantiles value must be normalized near one"
        )
    return (
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        float(quantiles[3]),
        float(quantiles[4]),
    )


def _canonical_payload(
    *,
    mode: str,
    residual_scale: float,
    field_scale_quantiles: tuple[float, float, float, float, float],
    owns_field_scale: bool,
    source_release: str,
    estimator_version: str,
) -> dict[str, Any]:
    return {
        "kind": VISNoiseCalibration.KIND,
        "version": VISNoiseCalibration.VERSION,
        "mode": mode,
        "residual_scale": residual_scale,
        "field_scale_quantiles": list(field_scale_quantiles),
        "owns_field_scale": owns_field_scale,
        "source_release": source_release,
        "estimator_version": estimator_version,
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VISNoiseCalibration:
    """Runtime VIS amplitude model with a verified SHA-256 identity."""

    KIND: ClassVar[str] = "euclid_mer_vis_noise"
    VERSION: ClassVar[int] = 2
    MODE: ClassVar[str] = "amplitude_only"
    FIELD_SCALE_PROBABILITIES: ClassVar[tuple[float, ...]] = (
        0.0,
        0.16,
        0.5,
        0.84,
        1.0,
    )

    mode: str
    residual_scale: float
    field_scale_quantiles: tuple[float, float, float, float, float]
    owns_field_scale: bool
    source_release: str
    estimator_version: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.mode != self.MODE:
            raise ValueError(
                f"VIS noise calibration mode must be {self.MODE!r}"
            )

        if isinstance(self.residual_scale, bool):
            raise TypeError("residual_scale must be a finite positive number")
        residual_scale = float(self.residual_scale)
        if not np.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("residual_scale must be a finite positive number")
        object.__setattr__(self, "residual_scale", residual_scale)

        field_scale_quantiles = _validated_field_scale_quantiles(
            self.field_scale_quantiles
        )
        object.__setattr__(
            self,
            "field_scale_quantiles",
            field_scale_quantiles,
        )

        if type(self.owns_field_scale) is not bool:
            raise TypeError("owns_field_scale must be a bool")
        if not isinstance(self.source_release, str) or not self.source_release.strip():
            raise ValueError("source_release must be a non-empty string")
        if (
            not isinstance(self.estimator_version, str)
            or not self.estimator_version.strip()
        ):
            raise ValueError("estimator_version must be a non-empty string")
        if not isinstance(self.fingerprint, str) or not _FINGERPRINT_RE.fullmatch(
            self.fingerprint
        ):
            raise ValueError("fingerprint must be 64 lowercase hexadecimal characters")

        expected = _fingerprint(self._payload_without_fingerprint())
        if self.fingerprint != expected:
            raise ValueError("VIS noise calibration fingerprint does not match payload")

    @classmethod
    def build(
        cls,
        *,
        residual_scale: float,
        field_scale_quantiles: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        owns_field_scale: bool = True,
        source_release: str,
        estimator_version: str,
    ) -> VISNoiseCalibration:
        """Validate behavior fields and mint their canonical fingerprint."""
        if isinstance(residual_scale, bool):
            raise TypeError("residual_scale must be a finite positive number")
        scale = float(residual_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("residual_scale must be a finite positive number")
        quantiles = _validated_field_scale_quantiles(field_scale_quantiles)
        if type(owns_field_scale) is not bool:
            raise TypeError("owns_field_scale must be a bool")
        if not isinstance(source_release, str) or not source_release.strip():
            raise ValueError("source_release must be a non-empty string")
        if not isinstance(estimator_version, str) or not estimator_version.strip():
            raise ValueError("estimator_version must be a non-empty string")

        payload = _canonical_payload(
            mode=cls.MODE,
            residual_scale=scale,
            field_scale_quantiles=quantiles,
            owns_field_scale=owns_field_scale,
            source_release=source_release,
            estimator_version=estimator_version,
        )
        return cls(
            mode=cls.MODE,
            residual_scale=scale,
            field_scale_quantiles=quantiles,
            owns_field_scale=owns_field_scale,
            source_release=source_release,
            estimator_version=estimator_version,
            fingerprint=_fingerprint(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VISNoiseCalibration:
        """Load a strict amplitude-only payload and verify its fingerprint."""
        if not isinstance(payload, Mapping):
            raise TypeError("VIS noise calibration payload must be a mapping")
        expected_keys = {
            "kind",
            "version",
            "mode",
            "residual_scale",
            "field_scale_quantiles",
            "owns_field_scale",
            "source_release",
            "estimator_version",
            "fingerprint",
        }
        keys = set(payload)
        if keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(
                "invalid VIS noise calibration schema; "
                f"missing={missing}, extra={extra}"
            )
        if payload["kind"] != cls.KIND:
            raise ValueError(
                f"VIS noise calibration kind must be {cls.KIND!r}"
            )
        if type(payload["version"]) is not int or payload["version"] != cls.VERSION:
            raise ValueError(
                f"VIS noise calibration version must be {cls.VERSION}"
            )
        return cls(
            mode=payload["mode"],
            residual_scale=payload["residual_scale"],
            field_scale_quantiles=_validated_field_scale_quantiles(
                payload["field_scale_quantiles"]
            ),
            owns_field_scale=payload["owns_field_scale"],
            source_release=payload["source_release"],
            estimator_version=payload["estimator_version"],
            fingerprint=payload["fingerprint"],
        )

    def _payload_without_fingerprint(self) -> dict[str, Any]:
        return _canonical_payload(
            mode=self.mode,
            residual_scale=self.residual_scale,
            field_scale_quantiles=self.field_scale_quantiles,
            owns_field_scale=self.owns_field_scale,
            source_release=self.source_release,
            estimator_version=self.estimator_version,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable, self-verifying artifact payload."""
        return {**self._payload_without_fingerprint(), "fingerprint": self.fingerprint}

    def apply(
        self,
        residual: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Set one residual's absolute RMS without changing its structure.

        The input mean is restored after scaling.  This keeps
        any realized sky-subtraction offset while changing only its spatial
        fluctuations; callers add the result back to the untouched signal.
        ``residual_scale`` is the median absolute target robust RMS, not a
        multiplier on the detector model's pre-calibration RMS. When this model
        owns field scale and ``rng`` is supplied, one factor is drawn from the
        calibrated inverse CDF; ``rng=None`` uses its median. A model that does
        not own field scale uses factor one and leaves variation to its caller.
        """
        array = np.asarray(residual)
        if array.ndim != 2:
            raise ValueError(f"residual must be 2-D, got shape {array.shape}")
        if array.size == 0:
            raise ValueError("residual must be non-empty")
        if not np.all(np.isfinite(array)):
            raise ValueError("residual must contain only finite values")

        field_scale = 1.0
        if self.owns_field_scale:
            field_scale = self.field_scale_quantiles[2]
            if rng is not None:
                field_scale = float(np.interp(
                    float(rng.random()),
                    self.FIELD_SCALE_PROBABILITIES,
                    self.field_scale_quantiles,
                ))

        work = array.astype(np.float64, copy=False)
        input_mean = float(np.mean(work, dtype=np.float64))
        centered = work - input_mean
        centered_median = float(np.median(centered))
        input_scale = 1.4826 * float(
            np.median(np.abs(centered - centered_median))
        )
        if input_scale <= np.finfo(np.float64).eps:
            input_scale = float(np.sqrt(np.mean(centered * centered)))
        if input_scale <= np.finfo(np.float64).eps:
            # A constant residual has no stochastic structure to scale. Keep
            # its realized offset rather than manufacturing a random field.
            return np.full(array.shape, input_mean, dtype=np.float32)
        scaled = centered / input_scale
        # ``scaled`` is an affine transform of the original residual: no
        # convolution, resampling, padding, or neighbouring-pixel mixing.
        # Remove only floating-point DC drift before restoring the exact input
        # mean, preserving a realized sky-subtraction offset.
        scaled -= float(np.mean(scaled, dtype=np.float64))
        scaled *= self.residual_scale * field_scale
        scaled += input_mean
        return scaled.astype(np.float32)


__all__ = ["VISNoiseCalibration"]
