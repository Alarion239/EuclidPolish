"""11-23 meeting pass: every changed image in one multi-page PDF for a single Prezi upload ("Insert all pages"),
one JPEG per page embedded as is (DCTDecode, no re-encoding), page size = image size in points. The page order
is the manifest the Prezi replacement script reads (prezi_upload/v13_manifest.json: key per page)."""
import json
import os

from PIL import Image

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = os.path.join(T, "prezi_upload")
V = os.path.join(U, "v13")
pages = [  # (Prezi key, file)
    ("HERO_1", "HERO_1_euclid.jpg"), ("HERO_1b", "HERO_1b_euclid_from.jpg"), ("HERO_2", "HERO_2_polished.jpg"),
    ("HERO_3", "HERO_3_jwst.jpg"), ("CHART_base", "F1v6_wide_or_sharp_base.jpg"),
    ("CHART_euclid", "F1v6_wide_or_sharp_euclid.jpg"), ("SUMMARY", "SUMMARY_card.jpg"),
]
for n, name in ((1, "tidal"), (2, "barred2"), (7, "interacting"), (9, "inner_ring")):
    for k, lab in ((1, "Euclid"), (2, "POLISHed"), (3, "JWST")):
        pages.append((f"L{n:02d}{k}", f"L{n:02d}{k}_{name}_{lab}.jpg"))
for f in sorted(os.listdir(V)):
    if f.endswith(".jpg") and not f.startswith(("DR1_G6", "DR1_G11")):
        pages.append((f[:-4], os.path.join("v13", f)))


def pdf(items, out):
    objs, kids = [], []
    add = lambda b: objs.append(b) or len(objs)  # noqa: E731
    cat, pg = add(None), add(None)
    for _, f in items:
        path = os.path.join(U, f)
        w, h = Image.open(path).size
        data = open(path, "rb").read()
        img = add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                  b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (w, h, len(data))
                  + data + b"\nendstream")
        cs = b"q %d 0 0 %d 0 0 cm /Im0 Do Q" % (w, h)
        con = add(b"<< /Length %d >>\nstream\n" % len(cs) + cs + b"\nendstream")
        kids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %d %d] /Resources << /XObject << /Im0 %d 0 R >> >> "
                        b"/Contents %d 0 R >>" % (pg, w, h, img, con)))
    objs[cat - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pg
    objs[pg - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    body, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(body))
        body += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    body += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, cat, x)
    open(out, "wb").write(body)
    return len(body)


if __name__ == "__main__":
    out = os.path.join(T, "prezi_pdf", "v13_meeting.pdf")
    print(out, len(pages), "pages", round(pdf(pages, out) / 1e6, 2), "MB")
    json.dump([k for k, _ in pages], open(os.path.join(U, "v13_manifest.json"), "w"), indent=0)
