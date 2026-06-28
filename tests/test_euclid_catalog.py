"""Tests for euclid_polish.catalog.client.EuclidCatalog (astroquery mocked)."""
import pytest

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.client import EuclidAuthError, EuclidCatalog


class _FakeJob:
    def __init__(self, rows):
        self._rows = rows

    def get_results(self):
        return self._rows


class _FakeEuclid:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.logged_in = None

    def login(self, user=None, password=None):
        self.logged_in = (user, password)

    def launch_job_async(self, query):
        return _FakeJob(self.rows)

    def launch_job(self, query):
        return _FakeJob(self.rows)


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("EUCLID_USER", raising=False)
    monkeypatch.delenv("EUCLID_PASSWORD", raising=False)


def test_no_credentials_raises(no_env, monkeypatch):
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", _FakeEuclid())
    with pytest.raises(EuclidAuthError):
        EuclidCatalog()


def test_explicit_credentials_login(no_env, monkeypatch):
    fake = _FakeEuclid()
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", fake)
    EuclidCatalog(login="alice", password="pw")
    assert fake.logged_in == ("alice", "pw")


def test_env_credentials_login(monkeypatch):
    monkeypatch.setenv("EUCLID_USER", "bob")
    monkeypatch.setenv("EUCLID_PASSWORD", "secret")
    fake = _FakeEuclid()
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", fake)
    EuclidCatalog()
    assert fake.logged_in == ("bob", "secret")


def test_login_failure_raises(no_env, monkeypatch):
    class _Boom(_FakeEuclid):
        def login(self, **k):
            raise RuntimeError("bad creds")
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", _Boom())
    with pytest.raises(EuclidAuthError):
        EuclidCatalog(login="x", password="y")


def test_query_bright_stars_returns_objects(monkeypatch):
    rows = [{"right_ascension": 10.0, "declination": -5.0,
             "flux_vis_psf": 500.0, "fluxerr_vis_psf": 5.0},
            {"right_ascension": 11.0, "declination": -4.0,
             "flux_vis_psf": 300.0, "fluxerr_vis_psf": 4.0}]
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", _FakeEuclid(rows))
    cat = EuclidCatalog._unauthenticated()
    objs = cat.query_bright_stars(5)
    assert len(objs) == 2
    assert all(isinstance(o, CatalogObject) and o.kind == "star" for o in objs)
    assert objs[0].flux_psf_uJy == 500.0 and objs[0].magnitude is not None


def test_query_galaxies_returns_objects(monkeypatch):
    rows = [{"object_id": 7, "right_ascension": 10.0, "declination": -5.0,
             "segmentation_area": 200.0, "flux_vis_psf": 5000.0}]
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", _FakeEuclid(rows))
    cat = EuclidCatalog._unauthenticated()
    objs = cat.query_galaxies(10.0, -5.0, 0.05)
    assert len(objs) == 1 and objs[0].kind == "galaxy"


def test_unauthenticated_seam_skips_login(no_env, monkeypatch):
    fake = _FakeEuclid()
    monkeypatch.setattr("euclid_polish.catalog.client.Euclid", fake)
    EuclidCatalog._unauthenticated()
    assert fake.logged_in is None
