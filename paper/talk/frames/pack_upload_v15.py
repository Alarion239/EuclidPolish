"""11-25 storyboard grid: the two section-panel sizes and the 16:9 Next + QR tile in one PDF upload."""
import os

from pack_upload_v13 import T, pdf

pages = [("REGION_1", "REGION_1.jpg"), ("REGION_2", "REGION_2.jpg"), ("NEXT_tile", "NEXT_qr_tile.jpg")]
out = os.path.join(T, "prezi_pdf", "v15_storyboard.pdf")
print(out, round(pdf(pages, out) / 1e6, 2), "MB")
