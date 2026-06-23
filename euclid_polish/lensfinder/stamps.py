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


def cut_triplet(lr_cube, sr_arr, hr_raw, *, cx: float, cy: float, m: int
                ) -> Dict[str, np.ndarray]:
    """Source-centered VIS stamps from a field's LR cube, SR array and HR truth.

    Returns ``{"lr_vis": (m//2, m//2), "sr_vis": (m, m), "hr_vis": (m, m)}`` on
    their native grids (LR at half resolution). VIS is band 0 of any cube.
    """
    from euclid_polish.eval.synthetic_runner import crop_stamp

    lr_cube = np.asarray(lr_cube, dtype=np.float32)
    sr_arr = np.asarray(sr_arr, dtype=np.float32)
    hr_raw = np.asarray(hr_raw, dtype=np.float32)
    hr_vis = hr_raw[..., 0] if hr_raw.ndim == 3 else hr_raw
    sr_vis = sr_arr[..., 0] if sr_arr.ndim == 3 else sr_arr
    return {
        "lr_vis": crop_stamp(lr_cube[..., 0], cx=cx / 2.0, cy=cy / 2.0, m=m // 2),
        "sr_vis": crop_stamp(sr_vis, cx=cx, cy=cy, m=m),
        "hr_vis": crop_stamp(hr_vis, cx=cx, cy=cy, m=m),
    }


def lr_upsample_to_grid(plane) -> np.ndarray:
    """Nearest 2× upsample so an LR stamp lands on the SR/HR (M×M) grid.

    Same operation ``synthetic_runner`` uses for the LR-vs-HR PSNR comparison —
    no new information, only the pixel grid changes, so the LR head sees the
    same M×M canvas as SR/HR and the comparison is fair.
    """
    a = np.asarray(plane)
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def render_stamp_png(plane, out_png: str, *, asinh_scale: float,
                     size: int = 424) -> str:
    """Render a 2-D VIS stamp to a square 3-channel 8-bit PNG for Zoobot.

    Array-input twin of ``zoobot_morph.render_vis_png`` (same asinh+percentile
    stretch and bilinear resize), since here we hold numpy stamps, not FITS.
    """
    import os

    from PIL import Image

    from euclid_polish.eval.zoobot_morph import stretch_to_uint8

    u8 = stretch_to_uint8(np.asarray(plane, dtype=np.float32),
                          asinh_scale=asinh_scale)
    img = Image.fromarray(u8, mode="L").convert("RGB")
    if size and (img.width != size or img.height != size):
        img = img.resize((size, size), Image.BILINEAR)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img.save(out_png)
    return out_png


#: VIS stamp accessor + the grid each lands on, in catalog ``recon`` order.
def recon_planes(triplet: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map a ``cut_triplet`` result to render-ready M×M planes per reconstruction.

    LR is upsampled to the SR/HR grid so all three share one canvas.
    """
    return {
        "lr": lr_upsample_to_grid(triplet["lr_vis"]),
        "sr": triplet["sr_vis"],
        "hr": triplet["hr_vis"],
    }
