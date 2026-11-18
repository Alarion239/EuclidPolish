"""Copy the talk images into prezi_draft/images/ numbered in path order, matching prezi_draft/BUILD.md."""
import os
import shutil

from style import OUT, TALK

DEST = os.path.join(TALK, "prezi_draft", "images")
ORDER = [
    ("01_C1_barred_euclid", "C1_barred_1_euclid"),
    ("02_F1_wide_or_sharp", "F1_wide_or_sharp"),
    ("03_PSF_VIS_core10", "PSF_VIS_core10"), ("04_PSF_Y_core10", "PSF_Y_core10"),
    ("05_PSF_J_core10", "PSF_J_core10"), ("06_PSF_H_core10", "PSF_H_core10"),
] + [(f"{6 + k:02d}_PSF_VIS_region{k}", f"PSF_VIS_region{k}") for k in range(1, 10)] + [
    ("16_LENS_ring102042919_euclid", "LENS_ring102042919_1_euclid"),
    ("17_LENS_NGC6505_euclid", "LENS_NGC6505_1_euclid"),
    ("18_F9_tng_atlas_wall", "F9_tng_atlas_wall"),
    ("19_F10_dust_box", "F10_a_dust_box_117266"), ("20_F10_box_cut", "F10_b_box_cut_229934"),
    ("21_F10_slabs", "F10_c_slabs_580250"), ("22_F10_particle_grain", "F10_d_particle_grain_229934"),
    ("23_F11_truth", "F11_0_truth"), ("24_F11_psf", "F11_1_psf"), ("25_F11_noise_artifacts", "F11_3_full"),
] + [(f"{26 + i:02d}_F12_quiz_{L}_euclid", f"F12_quiz_{L}_1_euclid") for i, L in enumerate("ABCDEFGH")] + [
    ("34_F13_architecture", "F13_architecture"),
    ("35_F14_cat_photo", "F14_cat_0_photo"), ("36_F14_cat_truth", "F14_cat_1_truth"),
    ("37_F14_cat_euclid", "F14_cat_2_euclid"), ("38_F14_cat_polished", "F14_cat_3_polished"),
] + [(f"{39 + i:02d}_F15_quiz_{L}_polished", f"F15_quiz_{L}_2_polished") for i, L in enumerate("ABCDEFGH")] + [
    (f"{47 + i:02d}_F15_quiz_{L}_truth", f"F15_quiz_{L}_3_truth") for i, L in enumerate("BCEH")] + [
    ("51_C1_barred_polished", "C1_barred_2_polished"), ("52_C1_barred_jwst", "C1_barred_3_jwst"),
    ("53_STAR_euclid", "STAR_1_euclid"), ("54_STAR_polished", "STAR_2_polished"),
    ("55_LENS_ring102042919_polished", "LENS_ring102042919_2_polished"),
    ("56_LENS_NGC6505_polished", "LENS_NGC6505_2_polished"),
]
n = 57
for j in range(1, 11):
    stem = next(f[:-len("_1_euclid.png")] for f in sorted(os.listdir(OUT)) if f.startswith(f"JWST{j:02d}_")
                and f.endswith("_1_euclid.png"))
    for part in ("1_euclid", "2_polished", "3_jwst"):
        ORDER.append((f"{n:02d}_{stem}_{part}", f"{stem}_{part}"))
        n += 1
EXTRA = [("90_extra_HUDF_area_euclid", "HUDF_area_1_euclid"), ("91_extra_HUDF_area_polished", "HUDF_area_2_polished"),
         ("92_extra_HUDF_zoom_euclid", "HUDF_zoom_1_euclid"), ("93_extra_HUDF_zoom_polished", "HUDF_zoom_2_polished"),
         ("94_extra_CATSEYE_euclid", "CATSEYE_1_euclid"), ("95_extra_CATSEYE_polished", "CATSEYE_2_polished"),
         ("96_extra_ring_wide_euclid", "LENS_ring102042919_wide_1_euclid"),
         ("97_extra_ring_wide_polished", "LENS_ring102042919_wide_2_polished"),
         ("98_extra_NGC6505_wide_euclid", "LENS_NGC6505_wide_1_euclid"),
         ("99_extra_NGC6505_wide_polished", "LENS_NGC6505_wide_2_polished")]
os.makedirs(DEST, exist_ok=True)
missing = []
for dst, src in ORDER + EXTRA:
    s = os.path.join(OUT, src + ".png")
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(DEST, dst + ".png"))
    else:
        missing.append(src)
shutil.copy2(os.path.join(OUT, "F12_quiz_answer_key.json"), os.path.join(DEST, "F12_quiz_answer_key.json"))
print(len(ORDER) + len(EXTRA) - len(missing), "copied; missing:", missing)
