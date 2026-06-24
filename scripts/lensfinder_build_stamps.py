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
    from tqdm import tqdm

    from euclid_polish.sky.source_catalog import read_sources
    from euclid_polish.sky.tfrecord import read_multiband_skyimages, tfrecord_path

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

    window = args.max_fields or 1_000_000
    lr_by = {r.index: r for r in read_multiband_skyimages(
        tfrecord_path(rdir, f"dirty_{args.subset}"), num_images=window)}
    sr_by = {r.index: r for r in read_multiband_skyimages(
        tfrecord_path(rdir, f"sr_{args.subset}"), num_images=window)}
    hr_by = {r.index: r for r in read_multiband_skyimages(
        tfrecord_path(rdir, f"hr_{args.subset}"), num_images=window)}
    common = sorted(set(lr_by) & set(sr_by) & set(hr_by) & set(by_field))
    if args.max_fields:
        common = common[:args.max_fields]
    if not common:
        print("no fields with matching LR/SR/HR + sources")
        return 1
    field = int(np.asarray(hr_by[common[0]].data, np.float32).shape[0])
    print(f"{len(common)} fields, HR field {field}px, stamp {m}px HR (LR {m // 2})")

    reporter.set_stage(f"cutting 4-band stamps from {len(common)} fields")
    rows = []
    n_lens = n_gal = 0
    for i, idx in enumerate(tqdm(common, desc="fields")):
        reporter.set_step(i, len(common), f"field {idx}")
        lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)
        sr_cube = np.asarray(sr_by[idx].data, dtype=np.float32)
        hr_cube = np.asarray(hr_by[idx].data, dtype=np.float32)

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
        del sr_cube, hr_cube, lr_cube

    reporter.set_step(len(common), len(common), "done")
    lf_catalog.assign_splits(rows, val_frac=args.val_frac,
                             test_frac=args.test_frac, seed=args.seed)
    out_csv = os.path.join(args.out_dir, "catalog.csv")
    lf_catalog.write_catalog(out_csv, rows)
    reporter.metric({"fields": len(common), "lens": n_lens, "galaxy": n_gal,
                     "stamps": len(rows)})
    print(f"\n✓ {n_lens} lens + {n_gal} galaxy sources → {len(rows)} stamps "
          f"({len(rows)//3} per recon) → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
