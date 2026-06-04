"""hst routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from astropy.io import fits
from astropy.io import fits as _fits
from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_fetcher import _local_path_for
from euclid_polish.web.fasrc_fetcher import is_allowed_remote_path
from euclid_polish.web.fasrc_fetcher import list_remote_dir
from euclid_polish.web.fasrc_fetcher import run_remote_python
from flask import abort
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_file
from matplotlib.colors import AsinhNorm
from scipy.signal import fftconvolve
from typing import Any
from typing import Dict
from typing import List
import importlib.util
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from euclid_polish.web.helpers.fits_render import _render_fits_to_png
from euclid_polish.web.helpers.paths import _safe_relpath


def register(app):

    # ---------------- HST PSF page (mirrors /psfs but reads from FASRC) ----
    @app.route("/hst-psf")
    def hst_psf_page():
        cfg_loaded = fasrc_config.load()
        psf_remote_path = f"{cfg_loaded.data_dir}/hst_psf/F814W.fits"
        kernel_remote_path = f"{cfg_loaded.data_dir}/hst_psf/diff_kernel_VIS.fits"
        # Cheap directory listing — one find round-trip.
        ok, entries, list_err = list_remote_dir(
            f"{cfg_loaded.data_dir}/hst_psf",
            glob_pattern="*.fits",
        )
        files: List[Dict[str, Any]] = []
        for e in (entries or []):
            files.append({
                "name":         e["name"],
                "size_mb":      round(e["size"] / (1024 * 1024), 2),
                "mtime":        e["mtime"],
                "remote_path":  f"{cfg_loaded.data_dir}/hst_psf/{e['name']}",
            })
        files.sort(key=lambda d: d["name"])
        return render_template(
            "hst_psf.html",
            files=files,
            list_ok=ok,
            list_err=list_err,
            psf_remote_path=psf_remote_path,
            kernel_remote_path=kernel_remote_path,
            remote_dir=f"{cfg_loaded.data_dir}/hst_psf",
        )

    # ---------------- HST cutouts (per-star gallery on FASRC) -------------
    @app.route("/hst-cutouts")
    def hst_cutouts_page():
        cfg_loaded = fasrc_config.load()
        # Use the existing pagination knob from cutouts.html for symmetry.
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        per_page = 60
        ok, entries, list_err = list_remote_dir(
            f"{cfg_loaded.data_dir}/hst_stars",
            glob_pattern="star_*.fits",
            max_entries=2000,
        )
        # Newest tile run is the most interesting; sort by mtime desc.
        entries = sorted(entries or [], key=lambda e: -float(e["mtime"]))
        total = len(entries)
        n_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, n_pages)
        start = (page - 1) * per_page
        end   = start + per_page
        page_items = []
        for e in entries[start:end]:
            page_items.append({
                "name":         e["name"],
                "size_kb":      round(e["size"] / 1024, 1),
                "remote_path":  f"{cfg_loaded.data_dir}/hst_stars/{e['name']}",
            })
        return render_template(
            "hst_cutouts.html",
            files=page_items,
            total=total, page=page, n_pages=n_pages,
            list_ok=ok, list_err=list_err,
            remote_dir=f"{cfg_loaded.data_dir}/hst_stars",
        )

    @app.route("/hst-psf/preview.png")
    def hst_psf_preview_png():
        """Pull the F814W PSF + render an asinh PNG for the page header.

        The ``?force=1`` query arg bypasses the rsync cache — used by
        the Sync button after the user has rebuilt the PSF on FASRC.
        """
        cfg_loaded = fasrc_config.load()
        remote = f"{cfg_loaded.data_dir}/hst_psf/F814W.fits"
        force = request.args.get("force") in ("1", "true", "True")
        result = _fasrc_fetcher.fetch_one_file(remote, force=force)
        if not result.ok:
            abort(404)
        png = _render_fits_to_png(result.local_path, Config.BAND_VIS, size=320)
        # When forced, also disable HTTP-level caching so the browser
        # actually paints the freshly-rendered PNG instead of the one
        # the proxy/UA stored 5 min ago.
        max_age = 0 if force else 600
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=max_age)

    def _compute_validate_arrays():
        """Shared preprocessing for ``/hst-psf/validate.png`` and the
        ``A⊛H``-save endpoint.

        Returns ``(e, h, krn, a_conv_h, paths)`` where ``paths`` is a
        dict of the source FITS files used (for provenance headers).

        Aborts the request with 404 if any of the three on-disk files
        is missing — callers don't need to re-check.
        """

        # 1. Resolve local file paths.
        cfg = fasrc_config.load()
        hst_path = _local_path_for(f"{cfg.data_dir}/hst_psf/F814W.fits")
        krn_path = _local_path_for(f"{cfg.data_dir}/hst_psf/diff_kernel_VIS.fits")
        euc_path = os.path.join(Config.EUCLID_PSF_DIR, "euclid_psf_VIS.fits")
        for p in (hst_path, krn_path, euc_path):
            if not os.path.isfile(p):
                abort(404, description=f"missing FITS: {p}")

        # 2. Load script helpers via importlib so we apply the SAME
        # preprocessing the kernel solver did. (The script isn't on
        # the import path; cheap to load once per request.)
        repo_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        repo_root = os.path.dirname(repo_root)        # …/EuclidPolish
        script = os.path.join(repo_root, "scripts",
                              "fasrc_compute_differential_kernel.py")
        spec = importlib.util.spec_from_file_location("_fdk", script)
        fdk  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fdk)

        # 3. Load FITS arrays + pixel scales.
        def _read(path):
            with fits.open(path, memmap=False) as hdul:
                return (np.asarray(hdul[0].data, dtype=np.float64),
                        float(hdul[0].header.get("PIXSCALE", 0.05)))
        hst_raw, hst_scale = _read(hst_path)
        euc_raw, euc_scale = _read(euc_path)
        krn,     _         = _read(krn_path)

        # 4. Mirror the script's pipeline: resample → common-side crop
        # → renormalise → bg-subtract → border-zero. We **do not**
        # recenter here: ``compute_differential_kernel`` does its own
        # internal recentering AND undoes the net shift on the kernel,
        # so the saved A satisfies ``A ⊛ H_unrecentered = E_unrecentered``
        # — the contract every downstream caller relies on
        # (fasrc_generate_hst_tfrecords applies A to raw HST cutouts).
        # Validate against the same un-recentered arrays the kernel was
        # designed to work with; if rel.RMS comes out high, the kernel
        # IS bad (not just being tested against the wrong thing).
        common_side  = 511
        border_pixels = 10
        e = fdk._resample_to_hr_grid(euc_raw, euc_scale)
        h = fdk._resample_to_hr_grid(hst_raw, hst_scale)
        e = fdk._centre_crop_to(e, common_side)
        h = fdk._centre_crop_to(h, common_side)
        e = e / e.sum(); h = h / h.sum()
        e = fdk._bg_subtract_and_clip(e)
        h = fdk._bg_subtract_and_clip(h)
        e = fdk._zero_borders(e, border_pixels=border_pixels)
        h = fdk._zero_borders(h, border_pixels=border_pixels)

        # 5. Apply A to H (mode='same' gives back the input shape).
        a_conv_h = fftconvolve(h, krn, mode="same")

        return e, h, krn, a_conv_h, {
            "hst": hst_path, "kernel": krn_path, "euclid": euc_path,
        }

    @app.route("/hst-psf/validate.png")
    def hst_psf_validate_png():
        """Sanity-check the differential kernel: render ``E`` vs ``A⊛H``
        vs residual side-by-side.

        Loads three FITS files entirely from the *local* state — no
        FASRC round-trip:

          * ``A``: rsync'd diff kernel in the fetcher's cache.
          * ``H``: rsync'd HST F814W ePSF, same cache.
          * ``E``: Euclid VIS empirical PSF in ``data/euclid_psf/``.

        Runs the exact same preprocessing the kernel solver did
        (resample, common-side crop, bg-subtract, border-zero,
        renormalise, sub-pixel recenter) on H and E so the comparison
        is honest. Then ``A ⊛ H_cleaned`` is computed via fftconvolve
        and rendered alongside ``E_cleaned`` and their residual.
        """
        e, h, krn, a_conv_h, _ = _compute_validate_arrays()

        # 6. Scalar diagnostics so the user can read off the numbers
        # instead of squinting at the pictures.
        peak_e        = float(e.max())
        peak_a_h      = float(a_conv_h.max())
        flux_e        = float(e.sum())
        flux_a_h      = float(a_conv_h.sum())
        residual      = a_conv_h - e
        rms_residual  = float(np.sqrt((residual ** 2).mean()))
        rms_e         = float(np.sqrt((e ** 2).mean()))
        rel_rms       = rms_residual / rms_e if rms_e > 0 else float("nan")

        # 7. Render the 3-panel figure.

        # Shared asinh stretch for the two PSF panels — the linear scale
        # value is what controls how aggressively faint structure is
        # boosted. Pick something that brings out the spikes without
        # saturating the core; 1 % of the peak is a sensible default.
        asinh_scale = max(peak_e, peak_a_h) * 0.01
        psf_norm = AsinhNorm(
            linear_width=asinh_scale, vmin=0.0,
            vmax=max(peak_e, peak_a_h),
        )
        # Residual stretch: symmetric asinh around 0 with a divergent
        # colormap. AsinhNorm is sign-preserving, so the same stretch
        # behaviour applies in both halves. Use 1 % of max|res| as the
        # linear-width so faint structure shows up but the deepest
        # excursions don't saturate the colormap.
        r_lim = float(np.max(np.abs(residual)))
        if r_lim <= 0:
            r_lim = 1.0     # avoid /0 in AsinhNorm
        res_norm = AsinhNorm(
            linear_width=max(r_lim * 0.01, 1e-12),
            vmin=-r_lim, vmax=+r_lim,
        )

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=110)
        axes[0].imshow(e,        cmap="gray_r", norm=psf_norm, origin="lower",
                       interpolation="nearest")
        axes[0].set_title(f"E (target Euclid)\npeak={peak_e:.3e}, "
                          f"flux={flux_e:.4f}", fontsize=10)
        axes[1].imshow(a_conv_h, cmap="gray_r", norm=psf_norm, origin="lower",
                       interpolation="nearest")
        axes[1].set_title(f"A ⊛ H (kernel result)\npeak={peak_a_h:.3e}, "
                          f"flux={flux_a_h:.4f}", fontsize=10)
        axes[2].imshow(residual, cmap="RdBu_r", norm=res_norm, origin="lower",
                       interpolation="nearest")
        axes[2].set_title(f"A⊛H − E (asinh)\nmax|res|={r_lim:.3e}, "
                          f"RMS/RMS(E)={rel_rms:.3f}", fontsize=10)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        peak_ratio = peak_a_h / peak_e if peak_e > 0 else float("nan")
        fig.suptitle(
            f"Kernel sanity check — peak(A⊛H)/peak(E)={peak_ratio:.3f}  "
            f"flux(A⊛H)={flux_a_h:.4f}  rel.RMS residual={rel_rms:.3f}",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return send_file(io.BytesIO(buf.getvalue()),
                         mimetype="image/png", max_age=0)

    @app.route("/api/hst-psf/save-a-conv-h", methods=["POST"])
    def api_hst_psf_save_a_conv_h():
        """Compute A⊛H via the same chain validate.png uses and save it
        to a sibling FITS in the HST-PSF cache directory.

        Returns JSON with the inspect-relative path so the UI can build
        ``/inspect?fits=…`` and ``/inspect/download?fits=…`` links —
        bytewise inspection without re-uploading or saving by hand.

        Side-effect (intentional): the saved file lives under the FASRC
        cache, which is in ``_inspectable_roots`` already, so the
        inspector accepts it without further whitelist tweaks.
        """

        _, _, _, a_conv_h, paths = _compute_validate_arrays()

        cfg = fasrc_config.load()
        out_path = _local_path_for(
            f"{cfg.data_dir}/hst_psf/a_conv_h_validate.fits"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        hdu = _fits.PrimaryHDU(
            np.ascontiguousarray(a_conv_h, dtype=np.float32),
        )
        h = hdu.header
        h["OBJECT"]   = ("A conv H (validation)",
                         "kernel applied to HST F814W ePSF")
        h["EUCLBAND"] = ("VIS", "Euclid band the kernel maps INTO")
        h["HSTFILT"]  = ("F814W", "HST filter the kernel maps FROM")
        h["PIXSCALE"] = (Config.DEFAULT_PIXEL_SCALE, "arcsec / pix")
        h["BUNIT"]    = ("", "dimensionless (sums to ~1)")
        h["COMMENT"]  = "Generated by /api/hst-psf/save-a-conv-h"
        h["SRC_HST"]  = (os.path.basename(paths["hst"]),
                         "source HST ePSF FITS")
        h["SRC_KRN"]  = (os.path.basename(paths["kernel"]),
                         "source differential kernel FITS")
        h["SRC_EUC"]  = (os.path.basename(paths["euclid"]),
                         "source Euclid VIS PSF FITS")
        hdu.writeto(out_path, overwrite=True)

        rel = _safe_relpath(os.path.realpath(out_path))
        return jsonify({
            "ok":          True,
            "rel_path":    rel,
            "size_bytes":  int(os.path.getsize(out_path)),
            "shape":       list(a_conv_h.shape),
            "peak":        float(a_conv_h.max()),
            "sum":         float(a_conv_h.sum()),
        })

    @app.route("/api/hst-psf/sync", methods=["POST"])
    def api_hst_psf_sync():
        """Re-rsync the HST PSF + differential kernel from FASRC.

        Bypasses the fetcher's 5-minute cache (``force=True``) so a
        kernel you just rebuilt on FASRC shows up locally without
        waiting. Reports per-file status. Files that don't exist on
        FASRC are reported as failed individually but don't block the
        others.
        """
        cfg_loaded = fasrc_config.load()
        targets = {
            "psf":    f"{cfg_loaded.data_dir}/hst_psf/F814W.fits",
            "kernel": f"{cfg_loaded.data_dir}/hst_psf/diff_kernel_VIS.fits",
        }
        results: Dict[str, Dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True)
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
        return jsonify({"ok": any_ok, "files": results})

    @app.route("/hst-cutouts/preview.png")
    def hst_cutout_preview_png():
        """Pull one HST star cutout + render a thumbnail PNG.

        Same pattern as /cutout-image but the source FITS lives on FASRC,
        not locally. Respects the fetcher's 50 MB cap (star stamps are
        ~260 KB so this is never a concern for this endpoint).
        """
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        try:
            size = int(request.args.get("size", 140))
        except ValueError:
            size = 140
        if size < 16 or size > 1024:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok:
            abort(404)
        png = _render_fits_to_png(result.local_path, Config.BAND_VIS, size=size)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=3600)

    # ---------------- HST tiles inspector (header + random cutout) -------
    @app.route("/hst-tiles")
    def hst_tiles_page():
        cfg_loaded = fasrc_config.load()
        remote_dir = f"{cfg_loaded.data_dir}/hst_hlsp"
        # Depth 3 catches files at both the canonical flat layout AND
        # the in-progress scratch layout astroquery uses while a job
        # is still running: <hst_hlsp>/mastDownload/HLSP/<obs_id>/<file>.
        # Without this, the page would show 0 new tiles for the entire
        # duration of a multi-hour download, then jump to the full count
        # only when the post-download flatten runs.
        ok, entries, list_err = list_remote_dir(
            remote_dir, glob_pattern="hlsp_cosmos_*.fits",
            max_entries=200, max_depth=3,
        )
        # De-duplicate by basename, preferring the larger size (catches
        # the brief window where a file exists both flat and nested
        # mid-flatten — flat is the canonical copy when it's complete).
        best: Dict[str, Dict[str, Any]] = {}
        for e in (entries or []):
            name = e["name"]
            prev = best.get(name)
            if prev is None or e["size"] > prev["size"]:
                best[name] = e
        tiles = [
            {
                "name":        e["name"],
                "size_gb":     round(e["size"] / 1e9, 2),
                "mtime":       e["mtime"],
                "remote_path": f"{remote_dir}/{e['name']}",
            }
            for e in best.values()
        ]
        tiles.sort(key=lambda t: t["name"])
        return render_template(
            "hst_tiles.html",
            tiles=tiles, list_ok=ok, list_err=list_err,
            remote_dir=remote_dir,
        )

    @app.route("/fasrc/tile/header")
    def fasrc_tile_header():
        """JSON: full FITS header of one tile (no big transfer)."""
        path = request.args.get("path", "").strip()
        if not path or not is_allowed_remote_path(path):
            abort(400)
        rc, out, err = run_remote_python(
            "scripts/fasrc_inspect_tile.py",
            ["--path", path, "--mode", "header"],
            binary=False, timeout=20,
        )
        if rc != 0:
            return jsonify({"ok": False,
                            "error": err.strip() or out[:500]}), 502
        try:
            payload = json.loads(out)
        except json.JSONDecodeError as e:
            return jsonify({"ok": False,
                            "error": f"bad header JSON: {e}"}), 502
        payload["ok"] = True
        return jsonify(payload)

    @app.route("/fasrc/tile/cutout.png")
    def fasrc_tile_cutout_png():
        """Stream a single random PNG cutout from a remote tile.

        The remote script emits the PNG to stdout and a JSON sidecar
        (centre, sigma-clipped stats, blank flag) to stderr. We surface
        the stats via an ``X-Cutout-Stats`` response header so the page
        JS can display "median, σ, is-blank" next to the image — handy
        for diagnosing "why does this cutout look like a gradient?"
        cases (usually a bright star or empty tile-edge region).
        """
        path = request.args.get("path", "").strip()
        if not path or not is_allowed_remote_path(path):
            abort(400)
        try:
            size = int(request.args.get("size", 256))
            seed = int(request.args.get("seed", -1))
        except ValueError:
            abort(400)
        if size < 32 or size > 1024:
            abort(400)
        rc, out_bytes, err = run_remote_python(
            "scripts/fasrc_inspect_tile.py",
            ["--path", path, "--mode", "cutout",
             "--size", str(size), "--seed", str(seed)],
            binary=True, timeout=30,
        )
        if rc != 0 or not out_bytes:
            return jsonify({"ok": False,
                            "error": (err if isinstance(err, str) else err.decode(errors="replace")).strip() or "empty cutout"}), 502
        resp = send_file(
            io.BytesIO(out_bytes), mimetype="image/png", max_age=0,
        )
        # err contains the JSON sidecar; pass it through as a header so
        # the page JS can render it without a second SSH round-trip.
        if isinstance(err, str) and err.strip():
            resp.headers["X-Cutout-Stats"] = err.strip().splitlines()[-1]
        return resp
