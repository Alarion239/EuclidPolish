"""SR.fits outputs get a provenance stamp linking the producing model.

Tests the pure helper jobs_impl/catalog_runner delegate to (no download/model).
"""

from __future__ import annotations

import os

from astropy.io import fits

from euclid_polish.eval.sr_provenance import write_sr_provenance
from euclid_polish.provenance.checkpoint import write_checkpoint_provenance
from euclid_polish.provenance.fits import read_stamp_cards
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Format, Stamp
from euclid_polish.provenance.store import ProvStore


def test_write_sr_provenance_cards_and_artifact(tmp_path):
    ckpt = tmp_path / "ckpt"
    write_checkpoint_provenance(str(ckpt), Stamp(id=ProvId("2f9c81aa")))
    store = ProvStore(str(tmp_path / "store"))
    sr_path = str(tmp_path / "obj" / "SR.fits")
    os.makedirs(os.path.dirname(sr_path), exist_ok=True)
    hdr = fits.Header()

    stamp = write_sr_provenance(hdr, checkpoint_dir=str(ckpt), sr_fits_path=sr_path,
                          store=store, git=None, descriptors={"ra": 10.0})
    assert stamp is not None

    back = read_stamp_cards(hdr)
    assert back.id == stamp.id
    assert ProvId("2f9c81aa") in back.parents        # the model is a parent

    art = store.get(stamp.id)
    assert art.format is Format.FITS
    assert art.path == sr_path
    assert art.descriptors["ra"] == 10.0


def test_write_sr_provenance_checkpoint_without_model_has_no_parent(tmp_path):
    """No checkpoint sidecar → SR still stamped, just with no model parent."""
    store = ProvStore(str(tmp_path / "store"))
    sr_path = str(tmp_path / "obj" / "SR.fits")
    os.makedirs(os.path.dirname(sr_path), exist_ok=True)
    hdr = fits.Header()
    stamp = write_sr_provenance(hdr, checkpoint_dir=str(tmp_path / "nockpt"),
                          sr_fits_path=sr_path, store=store, git=None)
    assert stamp is not None
    assert stamp.parents == ()           # unknown model → no parent
