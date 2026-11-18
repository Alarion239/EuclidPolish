"""Draft deck for the Hernquist-group talk, built from prezi_draft/images in path order, to import into Prezi
(Prezi turns each slide into a topic). Dark, one message per slide, speaker notes on every slide.

    python3 make_draft_deck.py      # system python with python-pptx 1.x
"""
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.dirname(HERE)
IMG = os.path.join(TALK, "prezi_draft", "images")
OUT = os.path.join(TALK, "prezi_draft", "EuclidPOLISH_Hernquist_draft.pptx")

BG = RGBColor(0x0F, 0x12, 0x18)
TXT = RGBColor(0xE6, 0xE8, 0xEB)
MUTED = RGBColor(0x8B, 0x93, 0xA1)
TEAL = RGBColor(0x2F, 0xD4, 0xBF)
CORAL = RGBColor(0xF0, 0x8A, 0x6C)
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
blank = prs.slide_layouts[6]


LIGHT = "--light" in sys.argv          # JPEG copies <= 900 px, so the deck stays under ~10 MB for upload
if LIGHT:
    OUT = OUT.replace(".pptx", "_light.pptx")


def img(name):
    matches = sorted(f for f in os.listdir(IMG) if f.endswith(".png") and name in f)
    if not matches:
        raise FileNotFoundError(name)
    path = os.path.join(IMG, matches[0])
    if not LIGHT:
        return path
    from PIL import Image as PILImage
    light = os.path.join(TALK, "prezi_draft", "light", matches[0][:-4] + ".jpg")
    if not os.path.exists(light):
        os.makedirs(os.path.dirname(light), exist_ok=True)
        im = PILImage.open(path).convert("RGB")
        im.thumbnail((900, 900))
        im.save(light, quality=82, optimize=True)
    return light


def new_slide(eyebrow, title, notes=""):
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    text(s, eyebrow.upper(), M, Inches(0.45), Inches(12), Inches(0.35), 12, TEAL, font="Courier New")
    text(s, title, M, Inches(0.8), Inches(12.1), Inches(0.8), 34, TXT, bold=True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(s, t, x, y, w, h, size, color, bold=False, font="Arial"):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(t.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
        p.space_after = Pt(6)
    return tb


def row(s, names, y, size, gap=Inches(0.25), x0=None):
    """Square images in a centred row."""
    n = len(names)
    total = n * size + (n - 1) * gap
    x = x0 if x0 is not None else int((W - total) / 2)
    for k, nm in enumerate(names):
        s.shapes.add_picture(img(nm), int(x + k * (size + gap)), int(y), size, size)


def body(s, t, y=Inches(6.55), size=16, color=MUTED):
    text(s, t, M, y, Inches(12.1), Inches(0.7), size, color)


# ------------------------------------------------------------------ 0. title
s = prs.slides.add_slide(blank)
s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
text(s, "HERNQUIST GROUP · 4 DEC 2026", M, Inches(1.6), Inches(7), Inches(0.4), 13, TEAL, font="Courier New")
text(s, "EuclidPOLISH", M, Inches(2.1), Inches(7), Inches(1.0), 54, TXT, bold=True)
text(s, "Super-resolving Euclid with a network\ntrained on TNG50 skies", M, Inches(3.15), Inches(6.6),
     Inches(1.2), 24, TXT)
text(s, "Aleksandr Belotserkovtsev · Harvard College\nwith Tania Barone and Liam Connor", M, Inches(4.75),
     Inches(6.6), Inches(0.9), 16, MUTED)
s.shapes.add_picture(img("01_C1_barred_euclid"), Inches(7.6), Inches(1.05), Inches(5.3), Inches(5.3))
s.notes_slide.notes_text_frame.text = ("Open on this galaxy: one barred spiral as Euclid sees it. "
                                       "Remember it; we come back to it at the end.")

# ------------------------------------------------------------------ 1. wide or sharp
s = new_slide("01 · introduction", "Wide surveys have always been ground-based",
              "SDSS, DES, HSC, now Rubin LSST: each wider and deeper, all blurred by the atmosphere to about "
              "0.6 arcsec at the best sites and 1 arcsec or more elsewhere. (Chart values are round numbers.)")
s.shapes.add_picture(img("02_F1_wide_or_sharp"), Inches(2.0), Inches(1.75), Inches(9.33), Inches(5.25))

s = new_slide("01 · introduction", "Space is sharp, but tiny",
              "Hubble and JWST image about ten square arcminutes at a time. COSMOS, the largest Hubble mosaic, "
              "covers 1.7 deg2. Space telescopes follow up on objects; they don't survey.")
s.shapes.add_picture(img("02_F1_wide_or_sharp"), Inches(2.0), Inches(1.75), Inches(9.33), Inches(5.25))

s = new_slide("01 · introduction", "Can we sharpen ground images instead?",
              "Luo+25 super-resolve SDSS/DECaLS to HSC quality; Akhaury+24 deconvolve ground images toward "
              "space resolution. The atmospheric PSF changes with time, position and wavelength, so the blur the "
              "network must undo is never known exactly, and there is no wide, sharp reference to check it.")
text(s, "Neural networks try (Luo+25, Akhaury+24).\n\nBut the atmosphere's blur changes every night,\n"
        "so the network never knows exactly what it is undoing,\nand nothing wide and sharp can check it.",
     M, Inches(2.2), Inches(11.5), Inches(3.5), 26, TXT)

s = new_slide("01 · introduction", "Radio did it first: POLISH",
              "An interferometer's PSF is set by its antenna layout, so it is known exactly. Connor+22 built "
              "POLISH: a WDSR network trained only on simulated skies; it beats CLEAN on PSNR/SSIM and deconvolves "
              "real VLA data in seconds. Wu+26 (POLISH++) add wide-field patches and an arcsinh stretch, and "
              "recover simulated lenses with image separations near the PSF: about 10x more lenses than CLEAN.")
text(s, "An interferometer's PSF is known exactly.\n\nPOLISH (Connor+22): trained only on simulations,\n"
        "it deconvolves real radio data.\n\nPOLISH++ (Wu+26): finds lenses near the PSF scale.",
     M, Inches(2.2), Inches(11.5), Inches(3.8), 26, TXT)
text(s, "[ add a POLISH figure here, credit Connor et al. 2022 ]", M, Inches(6.5), Inches(8), Inches(0.4), 12,
     MUTED, font="Courier New")

s = new_slide("01 · introduction", "Euclid: wide and a stable PSF",
              "Euclid's Wide Survey covers ~14,000 deg2 from space in VIS + three NISP bands. We measured the "
              "PSF from 17,905 Q1 stars; the nine sky regions on the right look alike, which is the stability. "
              "DR1-Foundation came out on 12 Nov 2026: 1,900 deg2, over a billion objects, ~15,000 strong-lens "
              "candidates. Roman launched in Aug 2026 and adds >5,000 deg2.")
row(s, ["03_PSF_VIS", "04_PSF_Y", "05_PSF_J", "06_PSF_H"], Inches(1.85), Inches(2.6), x0=M)
for k in range(9):
    s.shapes.add_picture(img(f"{7 + k:02d}_PSF_VIS_region"), int(M + (k % 9) * Inches(1.3)), Inches(4.75),
                         Inches(1.2), Inches(1.2))
body(s, "14,000 deg² · DR1-Foundation (12 Nov 2026): 1,900 deg², >1 billion objects, ~15,000 lens candidates",
     y=Inches(6.3))

s = new_slide("01 · introduction", "Why it matters: strong lenses",
              "Lenses weigh the deflector, dark matter included, reveal dark subhalos, give H0 from time delays, "
              "and magnify faint sources. Finding one needs the arc separated from the deflector, which fails when "
              "the Einstein radius approaches the PSF, so compact, lower-mass lenses are missed.")
row(s, ["16_LENS_ring102042919_euclid", "96_extra_ring_wide_euclid"], Inches(1.8), Inches(4.4))
body(s, "Compact lenses (Einstein radius ≈ PSF) are missed, so the known population is biased to massive deflectors.")

# ------------------------------------------------------------------ 2. method
s = new_slide("02 · method", "The truth comes from TNG50",
              "TNG50-SKIRT Atlas (Baes+24; Kovacic+25 in Euclid bands): 1,154 z = 0 galaxies, 10^9.8 to 10^12 "
              "Msun, five projections each, 100 pc/px. 48 shown, low to high stellar mass. Thank the room here.")
s.shapes.add_picture(img("18_F9_tng_atlas_wall"), Inches(3.4), Inches(1.75), Inches(6.53), Inches(4.9))
body(s, "TNG50-SKIRT Atlas · 1,154 galaxies at z = 0 · 100 pc/px · 1,074 used in training")

s = new_slide("02 · method", "What we don't trust, said plainly",
              "z = 0 only: we tried placing galaxies at z = 0.1-2.5 and dropped it; TNG now gives morphology only, "
              "colours and SFR come from a forest fitted to Q1. Faint light: 22-25% dropout pixels in the truth vs "
              "3.9% in JWST; the network paints that texture on real sky. SKIRT dust-octree facets (up to 36% of VIS "
              "light in subhalo 117266), periodic-box cuts (21 excluded), slabs. These are limits of how we use a "
              "z = 0 atlas, not complaints about TNG.")
row(s, ["19_F10", "20_F10", "21_F10", "22_F10"], Inches(1.85), Inches(2.75))
body(s, "z = 0 only · particle grain · dust-box facets · box cuts  →  morphology only, colours from Q1, soft edges, "
        "cut galaxies excluded", y=Inches(5.0))

s = new_slide("02 · method", "From truth to a Euclid image",
              "The real forward model on one synthetic field: measured empirical PSFs per band, 0.1 arcsec pixels, "
              "Q1 noise (median-combined stacks: for months we averaged and got the correlations wrong), cosmic "
              "rays, dead pixels, saturation.")
row(s, ["23_F11", "24_F11", "25_F11"], Inches(1.85), Inches(3.7), gap=Inches(0.35))
body(s, "measured PSFs · 0.1″ pixels · Q1 noise · artifacts", y=Inches(6.15))

s = new_slide("02 · method", "Quiz: which four are real?",
              "Let the room vote. Answer: A, D, F, G are real Q1 fields; B, C, E, H are synthetic.")
row(s, [f"F12_quiz_{L}_euclid" for L in "ABCD"], Inches(1.75), Inches(2.45), gap=Inches(0.2))
row(s, [f"F12_quiz_{L}_euclid" for L in "EFGH"], Inches(4.35), Inches(2.45), gap=Inches(0.2))

s = new_slide("02 · method", "The network",
              "POLISH's WDSR design, four bands in and out: eight arcsinh knees in, 16 residual blocks, a shared "
              "trunk plus a per-band path, a learned output knee. 15M training steps, 70 A100-hours; about "
              "0.4-0.5 dB per doubling of training. Zoom into panel (b), one residual block.")
s.shapes.add_picture(img("34_F13_architecture"), Inches(1.6), Inches(1.75), Inches(7.07), Inches(5.0))
text(s, "15M\ntraining steps\n\n70\nA100-hours", Inches(9.3), Inches(2.2), Inches(3.4), Inches(4), 28, TXT,
     bold=True)

s = new_slide("02 · method", "What did it learn? Put a cat through it",
              "A tabby photo as a 51-arcsec 'galaxy' through the same forward model. The POLISHed cat is sharper "
              "but not the photo: darker fur, higher contrast, its own texture. The prior is galaxies, which is "
              "why the TNG truth matters so much. Photo: Alvesgaspar, Wikimedia Commons, CC BY-SA 3.0.")
row(s, ["35_F14", "36_F14", "37_F14", "38_F14"], Inches(1.85), Inches(2.75))
body(s, "Sharper, but not a cat photo: its prior is galaxies.", y=Inches(5.1), size=20, color=TXT)
text(s, "Cat photo: Alvesgaspar, Wikimedia Commons, CC BY-SA 3.0", M, Inches(6.9), Inches(8), Inches(0.3), 10,
     MUTED)

# ------------------------------------------------------------------ 3. payoff
s = new_slide("03 · payoff", "Quiz, answered: everything POLISHed",
              "A, D, F, G were real. Every field POLISHed by the production model.")
row(s, [f"F15_quiz_{L}_polished" for L in "ABCD"], Inches(1.75), Inches(2.45), gap=Inches(0.2))
row(s, [f"F15_quiz_{L}_polished" for L in "EFGH"], Inches(4.35), Inches(2.45), gap=Inches(0.2))

s = new_slide("03 · payoff", "The synthetic ones, against their TNG truth",
              "Top: POLISHed synthetic fields B, C, E, H. Bottom: the clean truth. Stars are added by the forward "
              "model, so the truth has none.")
row(s, [f"F15_quiz_{L}_polished" for L in "BCEH"], Inches(1.75), Inches(2.45), gap=Inches(0.2))
row(s, [f"F15_quiz_{L}_truth" for L in "BCEH"], Inches(4.35), Inches(2.45), gap=Inches(0.2))

s = new_slide("03 · payoff", "The barred spiral, POLISHed",
              "Back to the opening galaxy, in colour with VIS included. The arm with its knots and the knot above "
              "are unclear in Euclid, clean in POLISHed, and JWST/NIRCam shows the same structure.")
row(s, ["01_C1_barred_euclid", "51_C1_barred_polished", "52_C1_barred_jwst"], Inches(1.75), Inches(3.7),
    gap=Inches(0.3))
body(s, "Same arm, same knots as JWST.", y=Inches(5.8), size=20, color=TXT)

s = new_slide("03 · payoff", "Stars become points",
              "A saturated Q1 star: spikes, halo and a blanked core are the telescope. The network deconvolves them "
              "and keeps the flux (VIS 21.7M e- in, 23.9M e- POLISHed), and faint galaxies under the halo appear.")
row(s, ["53_STAR_euclid", "54_STAR_polished"], Inches(1.75), Inches(4.4), gap=Inches(0.4))
body(s, "Spikes are the telescope, not the sky. The flux stays.", y=Inches(6.35), size=20, color=TXT)

s = new_slide("03 · payoff", "Lenses, POLISHed",
              "The grade-A Q1 lens from the introduction: Euclid shows a blob with hints of arcs; POLISHed shows a "
              "complete Einstein ring. No high-resolution truth exists for this one.")
row(s, ["16_LENS_ring102042919_euclid", "55_LENS_ring102042919_polished"], Inches(1.75), Inches(4.4),
    gap=Inches(0.4))

jw = sorted(f[:-4] for f in os.listdir(IMG) if "_JWST" in f and f.endswith("_1_euclid.png"))
for part in (jw[:5], jw[5:]):
    s = new_slide("03 · payoff", "Against JWST, galaxy by galaxy",
                  "Euclid | POLISHed | JWST/NIRCam, one calibration and stretch per row. NEXUS field, EDF-North.")
    for r, stem in enumerate(part):
        base = stem[:-len("_1_euclid")].split("_", 1)[1]          # drop the path-order number
        for k, suffix in enumerate(("_1_euclid", "_2_polished", "_3_jwst")):
            s.shapes.add_picture(img(base + suffix), int(Inches(2.6) + r * Inches(1.65)),
                                 int(Inches(1.75) + k * Inches(1.65)), Inches(1.55), Inches(1.55))
    text(s, "Euclid\n\n\nPOLISHed\n\n\nJWST", M, Inches(2.25), Inches(1.8), Inches(5), 14, MUTED)

s = new_slide("04 · limits", "What doesn't work yet, and a question for you",
              "Faint flux on real sky: 0.98 of the MER catalog at 21-23 mag, 0.90 at 24.5-25 (synthetic: 0.999). "
              "A faint pixel-grid pattern in the long run. Holes in very bright cores. All are sim-to-real; the "
              "synthetic test can't see them. Then ask the room.")
text(s, "Faint flux on real sky: 0.98 → 0.90 of the catalog from 21 to 25 mag\n"
        "A faint pixel-grid pattern · holes in very bright cores\nAll sim-to-real: the synthetic test can't see them",
     M, Inches(2.0), Inches(12), Inches(2.0), 22, TXT)
text(s, "If you were building the training truth from TNG,\nwhat would you change?", M, Inches(4.6),
     Inches(12), Inches(1.5), 32, CORAL, bold=True)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
