"""11-25 sidebar: the text-only Next tile and the portrait QR card in one PDF upload (page order: Next, QR)."""
import os

from pack_upload_v13 import T, pdf

pages = [("NEXT_text", "NEXT_text.jpg"), ("QR_portrait", "QR_portrait.jpg")]
out = os.path.join(T, "prezi_pdf", "v17_sidebar.pdf")
print(out, round(pdf(pages, out) / 1e6, 2), "MB")
