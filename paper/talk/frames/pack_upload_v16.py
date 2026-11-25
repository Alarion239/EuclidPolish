"""11-25 four-point layout: the three row-panel sizes in one PDF upload (page order: 5, 4, 2 tiles)."""
import os

from pack_upload_v13 import T, pdf

pages = [("ROW_5", "ROW_5.jpg"), ("ROW_4", "ROW_4.jpg"), ("ROW_2", "ROW_2.jpg")]
out = os.path.join(T, "prezi_pdf", "v16_rows.pdf")
print(out, round(pdf(pages, out) / 1e6, 2), "MB")
