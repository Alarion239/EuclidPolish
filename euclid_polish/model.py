"""The :class:`Model` operator — the public face of a trained checkpoint.

Wraps :func:`~euclid_polish.training.inference.load_model_from_checkpoint`
and :func:`~euclid_polish.training.inference.reconstruct`. This is the only
module that imports the inference engine for *upsampling*; cutout objects are
pure data and never import this module (enforced by a test).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from euclid_polish.config import Config
from euclid_polish.cutout.base import LRCutout, SRCutout
from euclid_polish.provenance.checkpoint import model_id_of_checkpoint
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.ids import ProvId
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.inference import (
    load_model_from_checkpoint as _default_load,
    reconstruct as _default_reconstruct,
)

_HR_SCALE = Config.DEFAULT_PIXEL_SCALE   # 0.05 arcsec/pix


class Model:
    """The public face of a trained WDSR checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the TF checkpoint directory.
    scale : int
        Super-resolution upscale factor (default 2).
    num_res_blocks : int
        Number of residual blocks (default ``Config.DEFAULT_NUM_RES_BLOCKS``).
    _load_fn, _reconstruct_fn : callable, optional
        Test injection points replacing ``load_model_from_checkpoint`` /
        ``reconstruct``.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        scale: int = 2,
        num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
        _load_fn: Optional[Callable] = None,
        _reconstruct_fn: Optional[Callable] = None,
    ) -> None:
        load_fn = _load_fn if _load_fn is not None else _default_load
        self._tf_model = load_fn(checkpoint_dir, scale, num_res_blocks)
        self._scale = int(scale)
        self._reconstruct_fn: Callable = (
            _reconstruct_fn if _reconstruct_fn is not None else _default_reconstruct
        )
        self.id: Optional[ProvId] = model_id_of_checkpoint(checkpoint_dir)

    def upsample_array(self, arr: np.ndarray) -> np.ndarray:
        """Super-resolve a bare numpy array (raw electrons); return the SR array.

        Input is 2D ``(H, W)`` or 3D ``(H, W, C)``. Output is
        ``(H·scale, W·scale)`` (single-output) or ``(H·scale, W·scale, C)``
        (4-band model).
        """
        _lr_display, sr_data = self._reconstruct_fn(self._tf_model, arr)
        return np.asarray(sr_data, dtype=np.float32)

    def upsample(self, lr: LRCutout, *, store=None) -> SRCutout:
        """Super-resolve an :class:`LRCutout` into an :class:`SRCutout`.

        The SR cutout's parents are ``(self.id, lr.id)`` (``self.id`` is
        omitted when ``None`` — a legacy checkpoint with no provenance
        sidecar). The new id is minted via ``store`` (or ``default_store()``).
        The provenance step is guarded so a store failure degrades to an
        unstamped but correct artifact.
        """
        _lr_display, sr_data = self._reconstruct_fn(self._tf_model, lr.image.data)
        sr_data = np.asarray(sr_data, dtype=np.float32)
        if sr_data.ndim == 3 and sr_data.shape[-1] == len(lr.band_names):
            bands = lr.band_names
        else:
            bands = ("VIS",)
        sr_img = MultiBandSkyImage(
            data=sr_data, pixel_scale_arcsec=_HR_SCALE, band_names=bands,
            is_clean=True, subset=lr.image.subset,
        )
        parents = tuple(p for p in (self.id, lr.id) if p is not None)
        try:
            _store = store if store is not None else default_store()
            new_id = _store.mint()
        except Exception:   # noqa: BLE001 — provenance is best-effort
            new_id = ProvId.sentinel()
        return SRCutout(image=sr_img, id=new_id, parents=parents)

    def eval_catalog(self, *args, **kwargs):
        """Evaluate the model on a lens catalog. Implemented in SP2."""
        raise NotImplementedError("eval_catalog is implemented in SP2")

    def eval_grouped(self, *args, **kwargs):
        """Grouped real-galaxy evaluation. Implemented in SP2."""
        raise NotImplementedError("eval_grouped is implemented in SP2")
