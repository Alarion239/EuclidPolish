"""11-25 pass: the dots-only chart (two versions) and the Next + big-QR card in one PDF upload (page order: base
chart, Euclid chart, card)."""
import os

from pack_upload_v13 import T, pdf

pages = [("CHART_base", "F1v7_wide_or_sharp_base.jpg"), ("CHART_euclid", "F1v7_wide_or_sharp_euclid.jpg"),
         ("NEXT_qr", "NEXT_qr_wide.jpg")]
out = os.path.join(T, "prezi_pdf", "v14_overview_qr.pdf")
print(out, round(pdf(pages, out) / 1e6, 2), "MB")
