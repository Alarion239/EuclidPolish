"""Pack the 11-22 (v6) Prezi additions into ONE upload PDF, pages in z-order:
1-7  the proportional-area ladder pages (hudf_area_ladder.py): DSS2, Pan-STARRS, DES, Hubble, JWST, Euclid, POLISHed
8    the radio card image (radio_panel.py)
9-18 JWST panels of the ten JWST-card galaxies, 19-28 their POLISHed panels (same framing as the Euclid tiles).
Uploaded as JPEG-recompressed (ghostscript pdfwrite /DCTEncode, no downsampling) and split under Prezi's
10 MB limit: DSS2 alone, then the rest. The ladder pages were later replaced by the two-page-per-survey
version of hudf_area_ladder.py (wide box + separate inset), uploaded the same way."""
import glob
import os
import subprocess
import tempfile

from PIL import Image

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I, U, P = (os.path.join(T, d) for d in ("images", "prezi_upload", "prezi_pdf"))
pages = [os.path.join(I, f"AL_{k}.pdf") for k in ("dss", "ps1", "ls", "hst", "jwst", "euclid", "polished")]
tmp = tempfile.mkdtemp()


def jpg_page(path):
    out = os.path.join(tmp, os.path.basename(path) + ".pdf")
    Image.open(path).convert("RGB").save(out, resolution=150, quality=92)
    return out


pages.append(jpg_page(os.path.join(U, "RADIO_dsa2000_dirty.jpg")))
for suffix in ("3_*_JWST", "2_*_POLISHed"):
    files = sorted(glob.glob(os.path.join(U, f"L[01][0-9]{suffix}.jpg")))
    assert len(files) == 10, files
    pages += [jpg_page(f) for f in files]
out = os.path.join(P, "v6_upload.pdf")
subprocess.run(["pdfunite", *pages, out], check=True)
print(out, len(pages), "pages", round(os.path.getsize(out) / 1e6, 1), "MB")
