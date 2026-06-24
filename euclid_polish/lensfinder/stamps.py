"""Cut source-centered LR/SR/HR stamps from simulated fields and render PNGs.

Reuses the existing source-centered crop machinery (``synthetic_runner``) and
Zoobot rendering (``zoobot_morph``). Heavy imports (numpy is fine at module
scope; astropy/PIL/TF are pulled lazily inside functions) so importing this
module never drags in PIL/astropy and never touches torch.

The geometry mirrors ``synthetic_runner.run_synthetic_eval``: source coords are
on the HR grid; the LR cube is half-resolution, so LR crops use
``(cx/2, cy/2, m//2)`` while SR/HR use ``(cx, cy, m)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def iter_field_sources(
    sources: Sequence[Dict[str, Any]], *,
    want_type: str, field: int, m: int, edge_margin: float = 0.0,
) -> List[Dict[str, Any]]:
    """Every source of ``want_type`` that fits an m×m stamp inside the field.

    Edge filter mirrors ``synthetic_runner.select_central_source``: a source
    must sit at least ``m/2 + edge_margin`` HR pixels from each border (so the
    full stamp is in-bounds). Unlike that function — which returns only the most
    central source — this returns ALL fitting sources, since the lens-finder
    wants every lens plus a sample of galaxies per field.
    """
    half = m / 2.0 + float(edge_margin)
    out: List[Dict[str, Any]] = []
    for s in sources:
        if s.get("type") != want_type:
            continue
        x, y = float(s["x_pix"]), float(s["y_pix"])
        if x < half or x > field - half or y < half or y > field - half:
            continue
        out.append(s)
    return out


def sample_galaxy_negatives(
    galaxies: Sequence[Dict[str, Any]], n_keep: int, *,
    rng: np.random.Generator, prefer_bright: bool = True,
) -> List[Dict[str, Any]]:
    """Pick up to ``n_keep`` galaxy negatives.

    ``prefer_bright`` keeps the highest VIS-flux galaxies (hard negatives that
    are actually visible above the noise); otherwise samples uniformly.
    """
    gals = list(galaxies)
    if n_keep >= len(gals):
        return gals
    if prefer_bright:
        gals.sort(key=lambda s: -float(s.get("flux_vis_e", 0.0) or 0.0))
        return gals[:n_keep]
    idx = rng.choice(len(gals), size=n_keep, replace=False)
    return [gals[int(i)] for i in idx]


def cut_triplet(lr_cube, sr_cube, hr_cube, *, cx: float, cy: float, m: int
                ) -> Dict[str, np.ndarray]:
    """Source-centered 4-band stamps, one per reconstruction.

    Returns ``{"lr": (m//2, m//2, C), "sr": (m, m, C), "hr": (m, m, C)}`` on
    their native grids — LR is half-resolution, so it crops at
    ``(cx/2, cy/2, m//2)`` while SR/HR crop at ``(cx, cy, m)`` (same sky FOV).
    All bands are kept (VIS, Y_E, J_E, H_E); the renderer composites them.
    """
    from euclid_polish.eval.synthetic_runner import crop_stamp

    def _cube_crop(cube, ccx, ccy, mm):
        cube = np.asarray(cube, dtype=np.float32)
        return np.stack([crop_stamp(cube[..., c], cx=ccx, cy=ccy, m=mm)
                         for c in range(cube.shape[-1])], axis=-1)

    return {
        "lr": _cube_crop(lr_cube, cx / 2.0, cy / 2.0, m // 2),
        "sr": _cube_crop(sr_cube, cx, cy, m),
        "hr": _cube_crop(hr_cube, cx, cy, m),
    }


def render_stamp_rgb(stamp4, out_png: str, *,
                     scale_r: float = 1.0, scale_g: float = 1.0,
                     scale_b: float = 1.0, stretch: float = 100.0,
                     Q: float = 8.0, size: int = 424) -> str:
    """Render a 4-band (H, W, 4) e- stamp to a Lupton-asinh RGB PNG.

    Band->channel matches Zoobot's GZ-DECaLS scheme (bluest->B, reddest->R):
    B=VIS (ch0), G=mean(Y_E, J_E) (ch1,2), R=H_E (ch3). ``scale_*`` are the
    per-band factors (GZ used 125/71/52 on fluxes); ``stretch``/``Q`` are the
    Lupton asinh parameters. Output is upscaled to ``size`` (bilinear)."""
    import os

    from astropy.visualization import make_lupton_rgb
    from PIL import Image

    a = np.asarray(stamp4, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] < 4:
        raise ValueError(f"render_stamp_rgb expects (H, W, >=4); got {a.shape}")
    b = a[..., 0] * scale_b
    g = 0.5 * (a[..., 1] + a[..., 2]) * scale_g
    r = a[..., 3] * scale_r
    rgb = make_lupton_rgb(r, g, b, stretch=stretch, Q=Q)   # (H, W, 3) uint8
    img = Image.fromarray(rgb, mode="RGB")
    if size and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.BILINEAR)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img.save(out_png)
    return out_png


def load_fits_cube(fits_path: str) -> np.ndarray:
    """Load a FITS primary HDU as a band-last float32 cube ``(H, W, C)``.

    Accepts band-first ``(C, H, W)`` cubes (the eval LR/SR/HR layout) and 2-D
    ``(H, W)`` planes (a legacy VIS-only ``HR.fits``), which become ``(H, W, 1)``.
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if data.ndim == 2:
        return data[..., None]
    if data.ndim == 3:
        return np.moveaxis(data, 0, -1)          # (C, H, W) → (H, W, C)
    raise ValueError(f"load_fits_cube expects 2-D or 3-D FITS; got {data.shape}")


def render_eval_stamp(fits_path: str, out_png: str, *, crop_m: int,
                      scale_r: float = 1.0, scale_g: float = 1.0,
                      scale_b: float = 1.0, stretch: float = 100.0,
                      Q: float = 8.0, size: int = 424) -> str:
    """Render an eval FITS cutout exactly as training renders a stamp.

    Mirrors ``lensfinder_build_stamps``: the eval cutouts are source-centered,
    so center-crop the cube to ``crop_m`` px (geometric center == source) and
    composite via :func:`render_stamp_rgb` with the same Lupton-asinh recipe.
    ``stretch`` defaults to ``Config.STRETCH_SCALE_E`` (100). A cube with fewer
    than four bands (e.g. a not-yet-regenerated VIS-only ``HR.fits``) is
    band-replicated to four so the render degrades to grayscale instead of
    crashing.
    """
    from euclid_polish.eval.synthetic_runner import crop_stamp

    cube = load_fits_cube(fits_path)
    h, w = cube.shape[:2]
    cropped = np.stack(
        [crop_stamp(cube[..., c], cx=w / 2.0, cy=h / 2.0, m=crop_m)
         for c in range(cube.shape[-1])], axis=-1)
    if cropped.shape[-1] < 4:
        reps = int(np.ceil(4 / cropped.shape[-1]))
        cropped = np.tile(cropped, (1, 1, reps))[..., :4]
    return render_stamp_rgb(cropped, out_png, scale_r=scale_r, scale_g=scale_g,
                            scale_b=scale_b, stretch=stretch, Q=Q, size=size)
