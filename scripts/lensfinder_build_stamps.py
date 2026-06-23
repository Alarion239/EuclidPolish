#!/usr/bin/env python
"""Build lens-finder training stamps from simulated fields (main TF env).

Reads simulated fields (``dirty_{subset}`` = LR, ``hr_{subset}`` = HR) plus the
source sidecar (``sources_{subset}.csv``), runs the SR model once per field, and
cuts source-centered LR/SR/HR VIS stamps around every lens and a sampled set of
galaxy negatives. Each stamp is rendered to a Zoobot PNG; one catalog CSV ties
them together (``is_lens``, ``theta_E_arcsec``, ``recon``, leakage-safe ``split``).

The catalog CSV is the cross-env contract: ``scripts/lensfinder_train.py`` (in
the EuclidPolishZoobot env) consumes it. SR inference (TensorFlow) lives here.

Usage (EuclidPolishEnv)::

    python scripts/lensfinder_build_stamps.py \
        --records-dir data/images/records_v2 --subset validate \
        --out-dir data/lensfinder/stamps --stamp-m 128 --neg-per-lens 2
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


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records-dir", required=True,
                   help="dir with dirty_/hr_{subset}.tfrecord + sources_{subset}.csv")
    p.add_argument("--subset", default="train",
                   help="record subset stem (train / validate)")
    p.add_argument("--out-dir", required=True, help="output dir for PNGs + catalog.csv")
    p.add_argument("--stamp-m", type=int, default=128,
                   help="stamp size in HR px (even); LR stamp is half this")
    p.add_argument("--neg-per-lens", type=int, default=2,
                   help="galaxy negatives kept per lens in a field")
    p.add_argument("--edge-margin", type=float, default=0.0,
                   help="extra HR-px margin a source must keep from the border")
    p.add_argument("--checkpoint", default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--asinh-scale", type=float, default=None)
    p.add_argument("--png-size", type=int, default=424)
    p.add_argument("--max-fields", type=int, default=0,
                   help="cap number of fields processed (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--prefer-bright-neg", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    import numpy as np
    from tqdm import tqdm

    from euclid_polish.sky.source_catalog import read_sources
    from euclid_polish.sky.tfrecord import read_multiband_skyimages, tfrecord_path
    from euclid_polish.training.inference import (load_model_from_checkpoint,
                                                  reconstruct)

    m = int(args.stamp_m)
    if m % 2:
        m += 1
    asinh = float(args.asinh_scale or Config.STRETCH_SCALE_E)
    rng = np.random.default_rng(args.seed)
    rdir = args.records_dir

    src_csv = os.path.join(rdir, f"sources_{args.subset}.csv")
    by_field = read_sources(src_csv)
    if not by_field:
        print(f"no sources in {src_csv}")
        return 1

    window = args.max_fields or 1_000_000
    lr_recs = read_multiband_skyimages(tfrecord_path(rdir, f"dirty_{args.subset}"),
                                       num_images=window)
    hr_recs = read_multiband_skyimages(tfrecord_path(rdir, f"hr_{args.subset}"),
                                       num_images=window)
    lr_by = {r.index: r for r in lr_recs}
    hr_by = {h.index: h for h in hr_recs}
    common = sorted(set(lr_by) & set(hr_by) & set(by_field))
    if args.max_fields:
        common = common[:args.max_fields]
    if not common:
        print("no fields with matching LR/HR + sources")
        return 1
    field = int(np.asarray(hr_recs[0].data, np.float32).shape[0])
    print(f"{len(common)} fields, HR field {field}px, stamp {m}px HR")

    model = load_model_from_checkpoint(
        args.checkpoint, Config.DEFAULT_REBIN_FACTOR, args.num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS)

    rows = []
    n_lens = n_gal = 0
    for idx in tqdm(common, desc="fields"):
        lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)
        hr_raw = np.asarray(hr_by[idx].data, dtype=np.float32)
        _, sr = reconstruct(model, lr_cube)
        sr = np.asarray(sr, dtype=np.float32)

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
                triplet = lf_stamps.cut_triplet(lr_cube, sr, hr_raw, cx=cx, cy=cy, m=m)
                planes = lf_stamps.recon_planes(triplet)
                base = f"{idx:05d}_{stype}_{j}"
                for recon, plane in planes.items():
                    png = os.path.join(args.out_dir, args.subset, recon, base + ".png")
                    lf_stamps.render_stamp_png(plane, png, asinh_scale=asinh,
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
        del sr, hr_raw, lr_cube

    lf_catalog.assign_splits(rows, val_frac=args.val_frac,
                             test_frac=args.test_frac, seed=args.seed)
    out_csv = os.path.join(args.out_dir, "catalog.csv")
    lf_catalog.write_catalog(out_csv, rows)
    print(f"\n✓ {n_lens} lens + {n_gal} galaxy sources → {len(rows)} stamps "
          f"({len(rows)//3} per recon) → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
