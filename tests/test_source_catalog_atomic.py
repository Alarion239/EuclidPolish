"""concat_source_csvs merges shard sidecars atomically and tolerates
sparse shards (a field that rendered no galaxies/lenses writes no row)."""

import os

from euclid_polish.sky.generation.source_catalog import SOURCE_COLS, concat_source_csvs


def _write_part(path, field_indices):
    with open(path, "w", newline="") as f:
        f.write(",".join(SOURCE_COLS) + "\r\n")
        for fi in field_indices:
            row = dict.fromkeys(SOURCE_COLS, "")
            row["field_index"] = str(fi)
            row["type"] = "galaxy"
            f.write(",".join(row[c] for c in SOURCE_COLS) + "\r\n")


def test_concat_merges_in_order_with_single_header(tmp_path):
    p0 = str(tmp_path / "sources_train.part0000.csv")
    p1 = str(tmp_path / "sources_train.part0001.csv")
    _write_part(p0, [0, 1])
    _write_part(p1, [])          # sparse shard: header only, no rows
    out = str(tmp_path / "sources_train.csv")

    concat_source_csvs([p0, p1], out)

    lines = [ln for ln in open(out).read().splitlines() if ln]
    assert lines[0] == ",".join(SOURCE_COLS)     # exactly one header
    assert sum(1 for ln in lines[1:]) == 2       # two data rows, sparse part ok
    assert ",".join(SOURCE_COLS) not in lines[1:]


def test_concat_leaves_no_temp_file(tmp_path):
    p0 = str(tmp_path / "sources_train.part0000.csv")
    _write_part(p0, [0])
    out = str(tmp_path / "sources_train.csv")

    concat_source_csvs([p0], out)

    leftovers = [n for n in os.listdir(tmp_path)
                 if n.startswith("sources_train.csv") and n != "sources_train.csv"]
    assert leftovers == []                       # temp file replaced, not left
    assert os.path.exists(out)
