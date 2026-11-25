"""11-25 picture pass: every re-rendered picture as Prezi uploads. JPEG pages (pack_upload_v13.pdf, page order =
prezi_upload/v18_manifest.json, split under Prezi's 10 MB upload limit) plus the vector survey-ladder pages
(ghostscript pdfwrite, images recompressed as JPEG, no downsampling)."""
import json
import os
import subprocess

from PIL import Image

from pack_upload_v13 import T, U, pdf

I, V = os.path.join(T, "images"), os.path.join(U, "v18")
os.makedirs(V, exist_ok=True)


def jpg(png, key, px=1600):
    out = os.path.join(V, key + ".jpg")
    Image.open(os.path.join(I, png)).convert("RGB").resize((px, px), Image.LANCZOS).save(out, quality=90)
    return os.path.relpath(out, U)


pages = [("HERO_3", "HERO_3_triptych.jpg"), ("CHART_base", "F1v8_wide_or_sharp_base.jpg"),
         ("CHART_euclid", "F1v8_wide_or_sharp_euclid.jpg")]
pages += [(f"PSF_{k}", jpg(f"PSF2_{k}.png", f"PSF_{k}")) for k in ("DES", "VIS", "H", "VIS_region1", "VIS_region2",
                                                                      "VIS_region3")]
pages += [(k, os.path.join("v13", k + ".jpg")) for k in ("A11_lens_ring_Euclid", "A12_NGC6505_ring_Euclid",
                                                          "K01_lens_ring_POLISHed", "K02_NGC6505_ring_POLISHed_optional")]
pages += [(f"DR1_G{g}_{e}", os.path.join("v13", f"DR1_G{g}_{e}.jpg")) for g in (2, 3, 4, 5) for e in "EP"]
pages += [("TNG_grid18", "v18/TNG_grid18.jpg")]
pages += [(f"C0{k}", jpg(f, f"C0{k}")) for k, f in enumerate(("F10_a_dust_box_117266.png", "F10_b_box_cut_229934.png",
                                                                "F10_c_slabs_580250.png", "F10_d_particle_grain_229934.png"), 1)]
pages += [(k, os.path.join("v13", k + ".jpg")) for k in ("D01_forward_truth", "D02_forward_PSF_pixels",
                                                          "D03_forward_noise_artifacts")]
pages += [(f"E0{i}", os.path.join("v13", f"E0{i}_quiz_{c}_Euclid.jpg")) for i, c in enumerate("ABCDEFGH", 1)]
pages += [("NET_simple", "v18/NET_simple.jpg")]
for n, name in ((1, "tidal"), (2, "barred2"), (7, "interacting"), (4, "edge_on")):
    for k, lab in ((1, "Euclid"), (2, "POLISHed"), (3, "JWST")):
        pages.append((f"L{n:02d}{k}", f"L{n:02d}{k}_{name}_{lab}.jpg"))
pages += [("JWST_grid", "v18/JWST_grid.jpg")]
pages += [("LIM3_euclid", jpg("LIM3_core_1_euclid.png", "LIM3_euclid")),
          ("LIM3_polished", jpg("LIM3_core_2_polished.png", "LIM3_polished"))]

parts, cur, size = [], [], 0
for key, f in pages:
    s = os.path.getsize(os.path.join(U, f))
    if cur and size + s > 8.5e6:
        parts.append(cur); cur, size = [], 0
    cur.append((key, f)); size += s
parts.append(cur)
man = []
for i, part in enumerate(parts, 1):
    out = os.path.join(T, "prezi_pdf", f"v18_{i}.pdf")
    print(out, len(part), "pages", round(pdf(part, out) / 1e6, 2), "MB")
    man.append([k for k, _ in part])
json.dump(man, open(os.path.join(U, "v18_manifest.json"), "w"), indent=0)

ladder = ["AL_hst_over_ls", "AL_jwst_area", "AL_euclid", "AL_euclid_inset", "AL_polished"]
out = os.path.join(T, "prezi_pdf", "v18_ladder.pdf")
subprocess.run(["gs", "-q", "-sDEVICE=pdfwrite", "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
                "-dDownsampleColorImages=false", "-dJPEGQ=92", "-o", out] + [os.path.join(I, f + ".pdf") for f in ladder],
               check=True)
print(out, ladder, round(os.path.getsize(out) / 1e6, 2), "MB")
