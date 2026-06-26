"""hstpairs routes for the EuclidPolish web UI (extracted from app.py).

EXPERIMENTAL — the HST Catalog page views the HST supervision lane's
TFRecord pairs, an experimental feature kept for future work and
disabled for now: ``register`` below is a no-op unless
``euclid_polish.web.experimental.EXPERIMENTAL_LANES_ENABLED`` is True.
"""
from __future__ import annotations

from astropy.io import fits
from euclid_polish.config import Config
from euclid_polish.image.tfio import read_multiband_skyimages
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.image import Image
from euclid_polish.web import experimental
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_fetcher import _local_path_for
from flask import abort
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_file
from scipy.signal import fftconvolve
from typing import Any
from typing import Dict
from typing import Optional
import io
import numpy as np
import os
from euclid_polish.web.helpers.fits_render import _arrays_to_fits_bytes
from euclid_polish.web.helpers.status import _record_count, _tfrecords_status


def register(app):
    # EXPERIMENTAL lane — disabled for now. Attribute read at call time
    # so tests can flip the flag before app creation.
    if not experimental.EXPERIMENTAL_LANES_ENABLED:
        return

    # =========================================================================
    # HST Catalog — same viewer as /sky but pointed at the FASRC-cached HST
    # TFRecord pairs that scripts/fasrc_generate_hst_tfrecords.py wrote.
    # =========================================================================

    def _hst_pairs_remote_dir() -> str:
        cfg = fasrc_config.load()
        return f"{cfg.data_dir}/images/records_v2_hst"

    def _hst_pairs_local_dir() -> str:
        """Local cache dir mirroring the remote HST records dir.

        Uses the same convention as :func:`fasrc_fetcher._local_path_for`
        so the page reads exactly what ``fetch_one_file`` writes — no
        chance of pointing at the wrong directory."""
        any_path = f"{_hst_pairs_remote_dir()}/clean_validate.tfrecord"
        return os.path.dirname(_local_path_for(any_path))

    @app.route("/hst-pairs")
    def hst_pairs_page():
        local_dir = _hst_pairs_local_dir()
        return render_template(
            "hst_pairs.html",
            tfrecords=_tfrecords_status(local_dir),
            local_dir=local_dir,
            remote_dir=_hst_pairs_remote_dir(),
            # Validation set is small (~80 MB per file × 3 files); train
            # set is multi-GB. Default the index nav to validate so the
            # page works the moment the user clicks Sync.
            default_subset="validate",
        )

    def _load_diff_kernel_local() -> Optional[np.ndarray]:
        """Return the analytic differential kernel from the local cache.

        Returns ``None`` (without raising) when the kernel FITS isn't
        synced yet, so callers can abort with a useful 404 message.
        """
        cfg = fasrc_config.load()
        krn_path = _local_path_for(
            f"{cfg.data_dir}/hst_psf/diff_kernel_VIS.fits"
        )
        if not os.path.isfile(krn_path):
            return None
        with fits.open(krn_path) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)

    def _hst_analytic_lr_cube(
        clean_hr_data: np.ndarray, kernel: np.ndarray, *, rebin: int = 2,
    ) -> np.ndarray:
        """Per-band ``fftconvolve(clean_HR, A)`` then sum-rebin × ``rebin``.

        Returns the noiseless deterministic forward at the LR grid —
        what a perfect forward model with no trained CNN and no noise
        injection would produce. Lets the user isolate ringing
        artifacts the CNN was meant to suppress.
        """
        out_hr = np.empty_like(clean_hr_data, dtype=np.float32)
        for c in range(clean_hr_data.shape[-1]):
            out_hr[..., c] = fftconvolve(
                clean_hr_data[..., c].astype(np.float32), kernel,
                mode="same",
            )
        return Image.rebin_array(out_hr, rebin)

    def _compute_hst_pair_arrays(subset, kind, index, records_dir):
        """Raw-array companion to ``_render_sky_record_png`` /
        ``_render_sky_record_pair_png``. Returns
        ``({name: array}, meta_dict)`` for the kind requested.

        For triptych ``kind="pair"`` the 4-band records keep their
        ``(H, W, C)`` shape so DS9 sees a proper data cube; FITS
        stores axes in NAXIS3-then-2 ordering so it surveys as
        "bands × H × W" via image-cube viewers.

        ``kind="dirty_analytic"`` is computed live from the matching
        ``clean`` record by applying the cached differential kernel
        (no CNN, no noise) — used by the live debug view to surface
        ringing artifacts directly attributable to the bare kernel.
        """
        if subset not in ("train", "validate"):
            abort(400)
        if kind not in ("clean", "dirty", "hr", "pair", "dirty_analytic"):
            abort(400)

        def _load(name):
            path = tfrecord_path(records_dir, name)
            if not os.path.exists(path):
                abort(404)
            recs = read_multiband_skyimages(path, num_images=max(index + 1, 1))
            if not recs or index >= len(recs):
                abort(404)
            return recs[min(index, len(recs) - 1)]

        meta_base = {
            "KIND":   kind,
            "SUBSET": subset,
            "INDEX":  int(index),
        }

        if kind == "pair":
            rs = {k: _load(f"{k}_{subset}") for k in ("clean", "dirty", "hr")}
            return (
                {k.upper(): np.asarray(rs[k].data, dtype=np.float32)
                 for k in ("clean", "dirty", "hr")},
                {**meta_base,
                 "PXSCALEC": float(rs["clean"].pixel_scale_arcsec),
                 "PXSCALED": float(rs["dirty"].pixel_scale_arcsec),
                 "PXSCALEH": float(rs["hr"].pixel_scale_arcsec)},
            )

        if kind == "dirty_analytic":
            clean = _load(f"clean_{subset}")
            kernel = _load_diff_kernel_local()
            if kernel is None:
                abort(404, description=(
                    "diff_kernel_VIS.fits not in local cache — run the "
                    "kernel step on FASRC and pull it down first."
                ))
            lr = _hst_analytic_lr_cube(
                np.asarray(clean.data, dtype=np.float32), kernel,
            )
            # HR pixel scale (0.05") halves to 0.10" after sum-rebin ×2.
            lr_pxscale = float(clean.pixel_scale_arcsec) * 2.0
            return (
                {"DIRTY_ANALYTIC": lr},
                {**meta_base, "PXSCALE": lr_pxscale,
                 "PXSCALEC": float(clean.pixel_scale_arcsec)},
            )

        rec = _load(f"{kind}_{subset}")
        return (
            {kind.upper(): np.asarray(rec.data, dtype=np.float32)},
            {**meta_base, "PXSCALE": float(rec.pixel_scale_arcsec)},
        )

    @app.route("/view/hst-pair.fits")
    def view_hst_pair_fits():
        """Raw-array FITS companion of ``/view/hst-pair``.

        Returns the linear electron pixel values for the requested
        kind. The ``band`` query arg controls slicing:

        * ``band`` in ``Config.LR_INPUT_BAND_NAMES`` (e.g. ``VIS``)
          → return only that band as a 2-D ``(H, W)`` image. Lets the
          user A/B one band against another viewer (e.g. compare
          ``dirty`` VIS to ``dirty_analytic`` VIS in DS9 without
          slicing).
        * ``band=color`` or absent → full ``(H, W, C)`` cube, same
          as before.
        * Single-band records (e.g. ``hr``) are returned as-is in
          either case.
        """
        subset = request.args.get("subset", "validate")
        kind   = request.args.get("kind",   "clean")
        band   = request.args.get("band",   "color")
        try:
            idx = int(request.args.get("i", 0))
        except ValueError:
            idx = 0
        arrays, meta = _compute_hst_pair_arrays(
            subset, kind, idx,
            records_dir=_hst_pairs_local_dir(),
        )
        # Slice down to one band when the chip selected a real band.
        # Triptych ``kind="pair"`` keeps all panels cubed because there
        # the user wants to inspect three records together; slicing
        # one out doesn't compose with the others.
        if (band in Config.LR_INPUT_BAND_NAMES
                and kind != "pair"):
            k_idx = list(Config.LR_INPUT_BAND_NAMES).index(band)
            sliced: Dict[str, np.ndarray] = {}
            for ext_name, arr in arrays.items():
                if arr.ndim == 3 and arr.shape[-1] > k_idx:
                    sliced[ext_name] = arr[..., k_idx]
                else:
                    # Single-band records (e.g. HR) pass through.
                    sliced[ext_name] = arr
            arrays = sliced
            meta = {**meta, "BAND": band}
        data = _arrays_to_fits_bytes(arrays, header_meta=meta)
        band_tag = f"_{band}" if band in Config.LR_INPUT_BAND_NAMES else ""
        fname = f"hst_{kind}_{subset}_{idx}{band_tag}.fits"
        return send_file(
            io.BytesIO(data), mimetype="application/fits",
            as_attachment=True, download_name=fname, max_age=0,
        )

    @app.route("/api/hst-pairs/totals")
    def api_hst_pairs_totals():
        local = _hst_pairs_local_dir()
        return jsonify({
            name: _record_count(name, records_dir=local)
            for name in ("clean_train", "clean_validate",
                         "dirty_train", "dirty_validate",
                         "hr_train",    "hr_validate")
        })

    @app.route("/api/hst-pairs/status")
    def api_hst_pairs_status():
        return jsonify(_tfrecords_status(_hst_pairs_local_dir()))

    @app.route("/api/hst-pairs/sync", methods=["POST"])
    def api_hst_pairs_sync():
        """Rsync HST TFRecord files from FASRC into the local cache.

        Form arg ``include_train`` (default false) controls whether the
        large train-split files are pulled. Validation files (~80 MB
        each, 3 files) are always included since they're small enough
        that "sync to view" makes sense; train can be many GB.

        Lifts the fetcher's default 50 MB cap to 5 GB per file because
        TFRecord files are intentionally large and the user explicitly
        asked for this transfer.
        """
        remote_dir = _hst_pairs_remote_dir()
        include_train = (request.values.get("include_train", "false")
                         .lower() in ("1", "true", "yes", "on"))
        targets: Dict[str, str] = {}
        for kind in ("clean", "dirty", "hr"):
            targets[f"{kind}_validate"] = (
                f"{remote_dir}/{kind}_validate.tfrecord"
            )
            if include_train:
                targets[f"{kind}_train"] = (
                    f"{remote_dir}/{kind}_train.tfrecord"
                )

        max_bytes = 5 * 1024 * 1024 * 1024
        results: Dict[str, Dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True, max_bytes=max_bytes)
            entry: Dict[str, Any] = {
                "remote_path": remote,
                "ok":          r.ok,
                "size_bytes":  r.size_bytes,
            }
            if r.ok and r.local_path:
                try:
                    entry["local_mtime"] = os.path.getmtime(r.local_path)
                except OSError:
                    entry["local_mtime"] = None
                any_ok = True
            else:
                entry["error"] = r.error
            results[key] = entry
        return jsonify({"ok": any_ok, "files": results,
                        "include_train": include_train})
