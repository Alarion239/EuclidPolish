"""11-27 (user, slides 8-9): on the returns to the survey window and to the sky-share card, Euclid on one
side and every other survey in turn on the other, so the audience sees Euclid is only beaten by Hubble/JWST
in sharpness and by the all-sky surveys in coverage.

WIN_<s>.jpg   the zoom window, 16:9: the same 25" x 28" field around the HUDF spiral, Euclid (left) | survey s
              (right), each at its native pixel and with the same colour as its inset on the first visit.
COV_<s>.jpg   the sky-share card below its title: two squares of the DSS2 all-sky box (3317"), dimmed, each with
              one survey's box (side ∝ sqrt(area)) filled with that survey's own wide image; Euclid left, s right.
              Needs data/al_cache/ from hudf_area_ladder.py.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

import hudf_area_ladder as AL
import hudf_ladder as L
from style import TALK

OUTD = os.path.join(TALK, "prezi_upload", "v24")
os.makedirs(OUTD, exist_ok=True)
FONT = os.path.join(TALK, "fonts", "SpaceGrotesk-Bold.ttf")
SURVEYS = ["dss", "ps1", "ls", "hst", "jwst"]
PSF = {k: v[2] for k, v in AL.NAMES.items()}
NAME = {k: v[0] for k, v in AL.NAMES.items()}
AREA = {k: v[1] for k, v in AL.NAMES.items()}

# ---------- window pairs ----------
FW, FH = 25.1, 28.35                       # field of one half, arcsec (two halves + gap = the 50.4" window)
GAP_PX, PX = 8, 0.025                      # output: 0.025"/px -> 2016 x 1134
CW, CH = int(round(FW / PX)), int(round(FH / PX))
CRA = AL.GRA - 0.6 / 3600 / AL.COSD        # the spiral, nudged 0.6" west so Euclid's field covers the left edge
CDEC = AL.GDEC


def sources(inst):
    """(planes as (array, wcs), native scale, smoothing sigma in px, stats region (ra, dec, size))."""
    if inst == "dss":
        a, w = L.load("dsswide_red.fits")
        a[a == 0] = np.nan
        return [(a, w)] * 3, 1.0, 0, (AL.WIN_RA, AL.WIN_DEC, 160.0)
    if inst == "ps1":
        return [L.load(f"ps1big_{b}.fits") for b in ("i", "r", "g")], 0.25, 1.0, (AL.WIN_RA, AL.WIN_DEC, 160.0)
    if inst == "ls":
        big, wb = L.load("lsbig_griz.fits")
        return [(big[i], wb) for i in (3, 1, 0)], 0.262, 0.7, (AL.WIN_RA, AL.WIN_DEC, 160.0)
    if inst == "euclid":
        return L.bands("euclid"), 0.1, 0, (L.RA0, L.DEC0, L.FIELD)
    return L.bands(inst), 0.03, 0, (AL.GRA, AL.GDEC, AL.SIDE[inst])


def half(inst):
    src, sc, sm, (sra, sdec, ssz) = sources(inst)
    stat = [AL.resample(a, w, sra, sdec, sc, ssz) for a, w in src]
    if sm:
        stat = [gaussian_filter(p, sm) for p in stat]
    med, white = AL.stretch(stat, inst)
    planes = [AL.resample(a, w, CRA, CDEC, sc, FW, FH) for a, w in src]
    if sm:
        planes = [gaussian_filter(p, sm) for p in planes]
    chroma = {"euclid": AL.CHROMA["euclid"] / 0.1, "ps1": AL.CHROMA["ground"] / sc, "ls": AL.CHROMA["ground"] / sc}
    img = AL.colour(planes, med, white, L.INST[inst][3], chroma_px=chroma.get(inst, 0), cap=inst == "euclid")
    im = Image.fromarray((img[::-1] * 255).astype(np.uint8))           # grid is y-up
    return im.resize((CW, CH), Image.LANCZOS if inst in ("hst", "jwst") else Image.NEAREST)


def label(d, xy, text, size, anchor="la"):
    f = ImageFont.truetype(FONT, size)
    d.text(xy, text, font=f, fill="white", stroke_width=max(2, size // 12), stroke_fill="black", anchor=anchor)


def window_pairs():
    eu = half("euclid")
    size = int(round(AL.WIN_LABEL / PX))
    pad = int(round(0.03 * FH / PX))
    for s in SURVEYS:
        canvas = Image.new("RGB", (2 * CW + GAP_PX, CH), "black")
        canvas.paste(eu, (0, 0))
        canvas.paste(half(s), (CW + GAP_PX, 0))
        d = ImageDraw.Draw(canvas)
        label(d, (pad, pad), f"Euclid, {PSF['euclid']}", size)
        label(d, (CW + GAP_PX + pad, pad), f"{NAME[s]}, {PSF[s]}", size)
        canvas.save(os.path.join(OUTD, f"WIN_{s}.jpg"), quality=92)
        print("WIN", s, canvas.size, flush=True)


# ---------- coverage pairs ----------
SQ, GAP2, W2, H2 = 940, 60, 2090, 1000     # px; the card's content area below the title is 17411 x 8331 units
DIM = 0.35


def box_square(inst, bg):
    side = AL.SIDE["dss"]
    sq = bg.copy() if inst != "dss" else None
    c = np.load(os.path.join(AL.CACHE_DIR, f"AL_cache_{inst}.npz"))
    if inst == "dss":
        sq = Image.fromarray((c["img"][::-1] * 255).astype(np.uint8)).resize((SQ, SQ), Image.LANCZOS)
        n = SQ
    else:
        n = max(2, int(round(AL.SIDE[inst] / side * SQ)))
        box = Image.fromarray((c["img"][::-1] * 255).astype(np.uint8)).resize((n, n), Image.LANCZOS)
        sq.paste(box, ((SQ - n) // 2, (SQ - n) // 2))
    d = ImageDraw.Draw(sq)
    x0 = (SQ - n) // 2
    d.rectangle([x0, x0, x0 + n - 1, x0 + n - 1], outline=(235, 235, 235), width=3 if n > 20 else 2)
    size, pad = 46, 14
    text = f"{NAME[inst]}, {AREA[inst]}"
    if n > 200:
        label(d, (x0 + pad, x0 + pad), text, size)
    else:                                       # a dot at this scale: label beside it
        label(d, (SQ // 2 + 14, SQ // 2 - 6), text, size, anchor="lb")
    return sq


def coverage_pairs():
    dss = np.load(os.path.join(AL.CACHE_DIR, "AL_cache_dss.npz"))["img"][::-1]
    bg = Image.fromarray((dss * DIM * 255).astype(np.uint8)).resize((SQ, SQ), Image.LANCZOS)
    eu = box_square("euclid", bg)
    x0 = (W2 - 2 * SQ - GAP2) // 2
    y0 = (H2 - SQ) // 2
    for s in SURVEYS:
        canvas = Image.new("RGB", (W2, H2), "black")   # black: the zoom window's layers sit over the gap
        canvas.paste(eu, (x0, y0))
        canvas.paste(box_square(s, bg), (x0 + SQ + GAP2, y0))
        canvas.save(os.path.join(OUTD, f"COV_{s}.jpg"), quality=92)
        print("COV", s, canvas.size, flush=True)


if __name__ == "__main__":
    coverage_pairs()
    window_pairs()
