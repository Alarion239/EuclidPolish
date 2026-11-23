"""Pack the 11-23 polish replacements into ONE upload PDF (pages in this order):
1-3  ladder insets with colour-only smoothing (hudf_area_ladder.py): Pan-STARRS, DES, Euclid
4    the wide-or-sharp chart with Euclid, its DR1 / full-survey points labelled (f1_survey_chart.py)
5-8  the cat card's four small tiles with readable labels (f14_cat.py): photo, truth, Euclid, POLISHed
9    the closing JWST hero frame with colour-only smoothing (hero_16x9.py)
In Prezi each page replaces the old object in place (same transform, z-order and path steps)."""
import os
import subprocess
import tempfile

from PIL import Image

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I, U, P = (os.path.join(T, d) for d in ("images", "prezi_upload", "prezi_pdf"))
tmp = tempfile.mkdtemp()


def img_page(path, dpi=150):
    out = os.path.join(tmp, os.path.basename(path) + ".pdf")
    Image.open(path).convert("RGB").save(out, resolution=dpi, quality=92)
    return out


pages = [os.path.join(I, f"AL_{k}_inset.pdf") for k in ("ps1", "ls", "euclid")]
pages.append(img_page(os.path.join(I, "F1v5_wide_or_sharp_euclid.png")))
pages += [img_page(os.path.join(I, f"F14_cat_{k}.png")) for k in ("0_photo", "1_truth", "2_euclid", "3_polished")]
pages.append(img_page(os.path.join(U, "HERO_3_jwst.jpg")))
out = os.path.join(P, "v12_polish.pdf")
subprocess.run(["pdfunite", *pages, out], check=True)
print(out, len(pages), "pages", round(os.path.getsize(out) / 1e6, 1), "MB")
