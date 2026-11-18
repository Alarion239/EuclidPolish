"""Standalone images for building the Prezi natively: one image per file, descriptive names in path
order, 1600 px JPEG (q 90). Individual TNG galaxies instead of the wall. Writes prezi_upload/."""
import glob
import json
import os
import shutil

from PIL import Image

from style import OUT, TALK

DEST = os.path.join(TALK, "prezi_upload")
os.makedirs(DEST, exist_ok=True)
for old in glob.glob(os.path.join(DEST, "*.jpg")):
    os.remove(old)
tng_dir = os.path.join(TALK, "data", "tng_wall")
meta = {m["id"]: m for m in json.load(open(os.path.join(tng_dir, "meta.json")))}
tng = sorted(glob.glob(os.path.join(tng_dir, "tng_*.png")),
             key=lambda f: meta[str(int(os.path.basename(f)[4:10]))]["logm"])
TNG_PICK = [3, 12, 15, 16, 17, 18, 24, 25, 28, 29, 36, 37]

items = [
    ("A01_title_barred_spiral_Euclid", "C1_barred_1_euclid"),
    ("A02_chart_wide_or_sharp", "F1_wide_or_sharp"),
    ("A03_PSF_VIS", "PSF_VIS_core10"), ("A04_PSF_NISP_Y", "PSF_Y_core10"),
    ("A05_PSF_NISP_J", "PSF_J_core10"), ("A06_PSF_NISP_H", "PSF_H_core10"),
    ("A07_PSF_VIS_wide_spikes", "PSF_VIS_wide40"),
    ("A08_PSF_VIS_skyregion1", "PSF_VIS_region1"), ("A09_PSF_VIS_skyregion2", "PSF_VIS_region2"),
    ("A10_PSF_VIS_skyregion3", "PSF_VIS_region3"),
    ("A11_lens_ring_Euclid", "LENS_ring102042919_1_euclid"),
    ("A12_NGC6505_ring_Euclid", "LENS_NGC6505_1_euclid"),
    ("A13_UltraDeepField_Euclid", "HUDF_area_1_euclid"),
]
items += [(f"B{k + 1:02d}_TNG50_galaxy_{os.path.basename(tng[i])[4:10]}", tng[i]) for k, i in enumerate(TNG_PICK)]
items += [
    ("C01_TNG_downside_dust_box", "F10_a_dust_box_117266"), ("C02_TNG_downside_box_cut", "F10_b_box_cut_229934"),
    ("C03_TNG_downside_slabs", "F10_c_slabs_580250"), ("C04_TNG_downside_particle_grain", "F10_d_particle_grain_229934"),
    ("D01_forward_truth", "F11_0_truth"), ("D02_forward_PSF_pixels", "F11_1_psf"),
    ("D03_forward_noise_artifacts", "F11_3_full"),
]
items += [(f"E{i + 1:02d}_quiz_{L}_Euclid", f"F12_quiz_{L}_1_euclid") for i, L in enumerate("ABCDEFGH")]
items += [
    ("F01_network_architecture", "F13_architecture"),
    ("G01_cat_photo", "F14_cat_0_photo"), ("G02_cat_truth", "F14_cat_1_truth"),
    ("G03_cat_Euclid", "F14_cat_2_euclid"), ("G04_cat_POLISHed", "F14_cat_3_polished"),
]
items += [(f"H{i + 1:02d}_quiz_{L}_POLISHed", f"F15_quiz_{L}_2_polished") for i, L in enumerate("ABCDEFGH")]
items += [(f"H{9 + i:02d}_quiz_{L}_TNGtruth", f"F15_quiz_{L}_3_truth") for i, L in enumerate("BCEH")]
items += [
    ("I01_barred_spiral_POLISHed", "C1_barred_2_polished"), ("I02_barred_spiral_JWST", "C1_barred_3_jwst"),
    ("J01_star_Euclid", "STAR_1_euclid"), ("J02_star_POLISHed", "STAR_2_polished"),
    ("K01_lens_ring_POLISHed", "LENS_ring102042919_2_polished"),
    ("K02_NGC6505_ring_POLISHed_optional", "LENS_NGC6505_2_polished"),
    ("K03_UltraDeepField_POLISHed", "HUDF_area_2_polished"),
    ("K04_UltraDeepField_zoom_Euclid", "HUDF_zoom_1_euclid"), ("K05_UltraDeepField_zoom_POLISHed", "HUDF_zoom_2_polished"),
]
jw = sorted({f.split("_1_euclid")[0] for f in os.listdir(OUT) if f.startswith("JWST") and "_1_euclid" in f})
for j, stem in enumerate(jw, 1):
    for part, lab in (("1_euclid", "Euclid"), ("2_polished", "POLISHed"), ("3_jwst", "JWST")):
        items.append((f"L{j:02d}{part[0]}_{stem.split('_', 1)[1]}_{lab}", f"{stem}_{part}"))

total = 0
for name, src in items:
    path = src if os.path.isabs(src) else os.path.join(OUT, src + ".png")
    im = Image.open(path).convert("RGB")
    im.thumbnail((1600, 1600))
    out = os.path.join(DEST, name + ".jpg")
    im.save(out, quality=90, optimize=True)
    total += os.path.getsize(out)
shutil.copy2(os.path.join(TALK, "data", "cat", "CREDIT.txt"), os.path.join(DEST, "CAT_PHOTO_CREDIT.txt"))
print(len(items), "images,", round(total / 1e6, 1), "MB")
