"""Pack the 11-22 Prezi additions into upload PDFs (one file per Prezi upload), with a page manifest."""
import json
import os

from PIL import Image

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I, U, P = (os.path.join(T, d) for d in ("images", "prezi_upload", "prezi_pdf"))
packs = {
    "v3a_replace": [("A01", "C1_barred_1_euclid"), ("A11", "LENS_ring102042919_1_euclid"), ("A12", "LENS_NGC6505_1_euclid"),
                    ("K04", "HUDF_zoom_1_euclid")] + [(f"E0{i + 1}", f"F12_quiz_{L}_1_euclid") for i, L in enumerate("ABCDEFGH")]
                   + [("CHART", "F1v2_wide_or_sharp"), ("LIM1", "LIM1_faint_flux")],
    "v3b_ladder": [(f"LF_{k}", f"HUDF_field_{k}") for k in ("dss", "ps1", "ls", "euclid", "polished")]
                  + [(f"LG_{k}", f"HUDF_galaxy_{k}") for k in ("dss", "ps1", "ls", "euclid", "polished", "hst", "jwst")],
    "v3c_dr1": [(f"DR1_G{k}_{s}", f"DR1L_G{k:02d}_{n}") for k in (2, 3, 4, 5, 6) for s, n in (("E", "1_euclid"), ("P", "2_polished"))],
    "v3d_zoom": [(os.path.splitext(f)[0], None) for f in sorted(os.listdir(U)) if f.startswith("Z_") and f.endswith(".jpg")],
}
man = {}
for name, items in packs.items():
    pages = []
    for key, stem in items:
        if stem is None:
            im = Image.open(os.path.join(U, key + ".jpg")).convert("RGB")
        else:
            im = Image.open(os.path.join(I, stem + ".png")).convert("RGB")
        w, h = im.size
        s = 1600 / max(w, h)
        pages.append(im.resize((round(w * s), round(h * s)), Image.LANCZOS))
    out = os.path.join(P, name + ".pdf")
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150, quality=90)
    man[name] = [k for k, _ in items]
    print(name, len(pages), round(os.path.getsize(out) / 1e6, 1), "MB")
json.dump(man, open(os.path.join(P, "v3_manifest.json"), "w"), indent=1)
