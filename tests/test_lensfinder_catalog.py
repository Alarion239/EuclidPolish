"""Tests for euclid_polish.lensfinder.catalog (pure, no torch/TF)."""

from __future__ import annotations

from euclid_polish.lensfinder import catalog as c


def _rows(n_fields=20, per_field=4):
    rows = []
    for fi in range(n_fields):
        for k in range(per_field):
            is_lens = 1 if k == 0 else 0          # one lens + 3 galaxies/field
            for recon in c.RECONS:
                rows.append({
                    "id_str": f"{fi:03d}_{k}_{recon}",
                    "file_loc": f"/tmp/{recon}/{fi}_{k}.png",
                    "is_lens": is_lens,
                    "theta_E_arcsec": (1.2 if is_lens else ""),
                    "recon": recon,
                    "field_index": fi,
                    "src_x_pix": 100.0, "src_y_pix": 100.0,
                })
    return rows


class TestCatalogIO:
    def test_round_trip(self, tmp_path):
        rows = _rows(3, 2)
        p = str(tmp_path / "catalog.csv")
        c.write_catalog(p, rows)
        back = c.read_catalog(p)
        assert len(back) == len(rows)
        assert back[0]["recon"] == rows[0]["recon"]
        assert back[0]["id_str"] == rows[0]["id_str"]

    def test_header_is_catalog_cols_first(self, tmp_path):
        p = str(tmp_path / "catalog.csv")
        c.write_catalog(p, _rows(1, 1))
        with open(p) as f:
            header = f.readline().strip().split(",")
        assert header[:len(c.CATALOG_COLS)] == c.CATALOG_COLS


class TestSplitsAndCounts:
    def test_assign_splits_is_field_grouped(self):
        rows = _rows(20, 4)
        c.assign_splits(rows, val_frac=0.2, test_frac=0.2, seed=0)
        # every field maps to exactly one split (no leakage)
        by_field = {}
        for r in rows:
            by_field.setdefault(r["field_index"], set()).add(r["split"])
        assert all(len(s) == 1 for s in by_field.values())

    def test_assign_splits_deterministic(self):
        a = c.assign_splits(_rows(20, 4), seed=7)
        b = c.assign_splits(_rows(20, 4), seed=7)
        assert [r["split"] for r in a] == [r["split"] for r in b]

    def test_subset_for_recon(self):
        rows = _rows(5, 4)
        sr = c.subset_for_recon(rows, "sr")
        assert sr and all(r["recon"] == "sr" for r in sr)
        assert len(sr) == len(rows) // 3

    def test_class_counts(self):
        rows = c.subset_for_recon(_rows(10, 4), "lr")   # 10 lens + 30 galaxies
        cc = c.class_counts(rows)
        assert cc == {"lens": 10, "nonlens": 30}
