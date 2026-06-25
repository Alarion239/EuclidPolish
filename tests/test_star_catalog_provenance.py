"""The star catalog gets a STABLE identity — minted once, reused across the
incremental saves of a download run."""

from __future__ import annotations

from euclid_polish.euclid.catalog import StarCatalog


def test_star_catalog_identity_is_stable(tmp_path):
    cat = StarCatalog(output_dir=str(tmp_path))
    cat.save({"stars": []})
    first = cat.prov_id()
    assert first is not None
    assert not first.is_sentinel

    cat.save({"stars": []})               # save again (incremental)
    assert cat.prov_id() == first         # same id, not a new one

    # A fresh handle over the same dir sees the same identity.
    assert StarCatalog(output_dir=str(tmp_path)).prov_id() == first


def test_star_catalog_without_save_has_no_identity(tmp_path):
    assert StarCatalog(output_dir=str(tmp_path)).prov_id() is None
