"""11-27 slide comments: the Euclid-vs-survey split images (survey_split.py), PSF / radio labels with commas
instead of middle dots, the limits core without hole marks, and the survey-ladder pages with comma labels.
JPEG pages -> prezi_pdf/v24_1.pdf (page order = v24_manifest.json); ladder pages (vector, gs) -> v24_ladder*.pdf."""
import json
import os
import subprocess

from PIL import Image

from pack_upload_v13 import T, U, pdf

I, V = os.path.join(T, "images"), os.path.join(U, "v24")


def jpg(png, key, px=1600):
    out = os.path.join(V, key + ".jpg")
    Image.open(os.path.join(I, png)).convert("RGB").resize((px, px), Image.LANCZOS).save(out, quality=90)
    return os.path.relpath(out, U)


S = ["dss", "ps1", "ls", "hst", "jwst"]
pages = [(f"WIN_{s}", f"v24/WIN_{s}.jpg") for s in S] + [(f"COV_{s}", f"v24/COV_{s}.jpg") for s in S]
pages += [(f"PSF_{k}", jpg(f"PSF2_{k}.png", f"PSF_{k}")) for k in ("DES", "VIS", "Y", "J", "H")]
pages += [(f"PSF_VIS_region{k}", jpg(f"PSF2_VIS_region{k}.png", f"PSF_VIS_region{k}")) for k in range(1, 6)]
pages += [("RADIO", "RADIO_dsa2000_dirty.jpg"), ("LIM3_polished", jpg("LIM3_core_2_polished.png", "LIM3_polished"))]
out = os.path.join(T, "prezi_pdf", "v24_1.pdf")
print(out, len(pages), "pages", round(pdf(pages, out) / 1e6, 2), "MB")
json.dump([k for k, _ in pages], open(os.path.join(U, "v24_manifest.json"), "w"), indent=0)

LADDER = [["AL_dss", "AL_ls", "AL_hst_over_ls", "AL_jwst_area", "AL_ls_inset", "AL_dss_inset", "AL_hst_area"], ["AL_ps1"]]
for i, group in enumerate(LADDER, 1):
    out = os.path.join(T, "prezi_pdf", f"v24_ladder{i}.pdf")
    subprocess.run(["gs", "-q", "-sDEVICE=pdfwrite", "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
                    "-dDownsampleColorImages=false", "-dJPEGQ=90", "-o", out] + [os.path.join(I, f + ".pdf") for f in group],
                   check=True)
    print(out, group, round(os.path.getsize(out) / 1e6, 2), "MB")
