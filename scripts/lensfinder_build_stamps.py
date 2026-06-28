#!/usr/bin/env python
"""Build lens-finder training stamps from simulated fields (main TF env, CPU).

Reads simulated fields (``dirty_{subset}`` = LR, ``sr_{subset}`` = SR,
``hr_{subset}`` = HR) plus the source sidecar (``sources_{subset}.csv``) and cuts
source-centered 4-band LR/SR/HR stamps around every lens and a sampled set of
galaxy negatives. Each stamp is rendered to a Lupton-asinh RGB PNG (B=VIS,
G=mean(Y,J), R=H); one catalog CSV ties them together (``is_lens``,
``theta_E_arcsec``, ``recon``, leakage-safe ``split``).

SR inference is NOT done here — the GPU ``scripts/lensfinder_sr_infer.py`` step
produces ``sr_{subset}.tfrecord`` first, so this step is pure crop/render/IO and
runs on the CPU ``shared`` partition. The catalog CSV is the cross-env contract:
``scripts/lensfinder_train.py`` (EuclidPolishZoobot env) consumes it.

Usage (EuclidPolishEnv)::

    python scripts/lensfinder_build_stamps.py \
        --records-dir data/images/records_lensfinder --subset validate \
        --out-dir data/lensfinder/stamps --stamp-m 106 --neg-per-lens 2
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.lensfinder import catalog as lf_catalog
from euclid_polish.lensfinder import stamps as lf_stamps
from euclid_polish.observability.reporter import Reporter


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records-dir", required=True,
                   help="dir with dirty_/hr_{subset}.tfrecord + sources_{subset}.csv")
    p.add_argument("--subset", default="train",
                   help="record subset stem (train / validate)")
    p.add_argument("--out-dir", required=True, help="output dir for PNGs + catalog.csv")
    p.add_argument("--stamp-m", type=int, default=106,
                   help="HR-grid stamp size (even); LR stamp is half (424/4=106)")
    p.add_argument("--neg-per-lens", type=int, default=2,
                   help="galaxy negatives kept per lens in a field")
    p.add_argument("--edge-margin", type=float, default=0.0,
                   help="extra HR-px margin a source must keep from the border")
    p.add_argument("--png-size", type=int, default=424,
                   help="encoder input size; stamps are upscaled to this")
    p.add_argument("--lupton-stretch", type=float, default=Config.STRETCH_SCALE_E)
    p.add_argument("--lupton-q", type=float, default=8.0)
    p.add_argument("--rgb-scale-r", type=float, default=1.0)
    p.add_argument("--rgb-scale-g", type=float, default=1.0)
    p.add_argument("--rgb-scale-b", type=float, default=1.0)
    p.add_argument("--max-fields", type=int, default=0,
                   help="cap number of fields processed (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--prefer-bright-neg", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()

    import numpy as np
    import tensorflow as tf
    from tqdm import tqdm

    from euclid_polish.sky.generation.source_catalog import read_sources
    from euclid_polish.image.tfio import tfrecord_path
    from euclid_polish.image import Image

    m = int(args.stamp_m)
    if m % 2:
        m += 1
    rng = np.random.default_rng(args.seed)
    rdir = args.records_dir

    src_csv = os.path.join(rdir, f"sources_{args.subset}.csv")
    by_field = read_sources(src_csv)
    if not by_field:
        print(f"no sources in {src_csv}")
        return 1

    paths = {k: tfrecord_path(rdir, f"{k}_{args.subset}")
             for k in ("dirty", "sr", "hr")}
    for k, p in paths.items():
        if not os.path.exists(p):
            print(f"missing {p}")
            return 1
    total = len(by_field)
    if args.max_fields:
        total = min(total, args.max_fields)

    # Stream the three field records in lockstep — never materialise every
    # field. A 2040²×4 SR/HR field is ~66 MB; 800 fields × (dirty+sr+hr) would
    # be ~120 GB and OOM. Peak memory here is one field's three cubes plus the
    # (small) catalog rows.
    ds = tf.data.Dataset.zip(tuple(
        tf.data.TFRecordDataset(paths[k]) for k in ("dirty", "sr", "hr")))

    reporter.set_stage(f"cutting 4-band stamps from {total} fields")
    rows = []
    n_lens = n_gal = 0
    field = None
    processed = 0
    for raw_lr, raw_sr, raw_hr in tqdm(ds, desc="fields", total=total):
        if args.max_fields and processed >= args.max_fields:
            break
        lr_img = Image.from_tfrecord(raw_lr)
        sr_img = Image.from_tfrecord(raw_sr)
        hr_img = Image.from_tfrecord(raw_hr)
        idx = lr_img.index
        if sr_img.index != idx or hr_img.index != idx:
            raise RuntimeError(
                f"record index misalignment: dirty={idx} sr={sr_img.index} "
                f"hr={hr_img.index} (dirty_/sr_/hr_ must share field order)")
        processed += 1
        reporter.set_step(processed, total, f"field {idx}")
        if idx not in by_field:
            continue
        lr_cube = np.asarray(lr_img.data, dtype=np.float32)
        sr_cube = np.asarray(sr_img.data, dtype=np.float32)
        hr_cube = np.asarray(hr_img.data, dtype=np.float32)
        if field is None:
            field = int(hr_cube.shape[0])
            print(f"HR field {field}px, stamp {m}px HR (LR {m // 2}); ~{total} fields")

        lenses = lf_stamps.iter_field_sources(
            by_field[idx], want_type="lens", field=field, m=m,
            edge_margin=args.edge_margin)
        galaxies = lf_stamps.iter_field_sources(
            by_field[idx], want_type="galaxy", field=field, m=m,
            edge_margin=args.edge_margin)
        negs = lf_stamps.sample_galaxy_negatives(
            galaxies, args.neg_per_lens * max(len(lenses), 1), rng=rng,
            prefer_bright=args.prefer_bright_neg)

        for stype, srcs in (("lens", lenses), ("galaxy", negs)):
            for j, s in enumerate(srcs):
                cx, cy = float(s["x_pix"]), float(s["y_pix"])
                triplet = lf_stamps.cut_triplet(lr_cube, sr_cube, hr_cube,
                                                cx=cx, cy=cy, m=m)
                base = f"{idx:05d}_{stype}_{j}"
                for recon, cube in triplet.items():
                    png = os.path.join(args.out_dir, args.subset, recon, base + ".png")
                    lf_stamps.render_stamp_rgb(
                        cube, png, scale_r=args.rgb_scale_r,
                        scale_g=args.rgb_scale_g, scale_b=args.rgb_scale_b,
                        stretch=args.lupton_stretch, Q=args.lupton_q,
                        size=args.png_size)
                    rows.append({
                        "id_str": f"{base}__{recon}",
                        "file_loc": png,
                        "is_lens": 1 if stype == "lens" else 0,
                        "theta_E_arcsec": (s.get("theta_E_arcsec", "")
                                           if stype == "lens" else ""),
                        "recon": recon,
                        "field_index": idx,
                        "src_x_pix": cx, "src_y_pix": cy,
                    })
                n_lens += stype == "lens"
                n_gal += stype == "galaxy"
        del lr_cube, sr_cube, hr_cube, lr_img, sr_img, hr_img

    if field is None:
        print("no fields with matching LR/SR/HR + sources")
        return 1
    reporter.set_step(total, total, "done")
    lf_catalog.assign_splits(rows, val_frac=args.val_frac,
                             test_frac=args.test_frac, seed=args.seed)
    out_csv = os.path.join(args.out_dir, "catalog.csv")
    lf_catalog.write_catalog(out_csv, rows)
    reporter.metric({"fields": processed, "lens": n_lens, "galaxy": n_gal,
                     "stamps": len(rows)})
    print(f"\n✓ {n_lens} lens + {n_gal} galaxy sources → {len(rows)} stamps "
          f"({len(rows)//3} per recon) → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
