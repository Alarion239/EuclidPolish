"""Laptop-side Euclid archive session for the web UI.

Holds the :class:`~euclid_polish.catalog.client.EuclidCatalog` the login form
(System › Connections, ``/auth/login``) created, so ``/auth/status`` can show
"logged in as <user>" and the local archive queries (galaxy / star
distributions, the catalog-eval galaxy query) can reuse it. The star-cutout
download runs on FASRC with its own credentials file (``/euclid-auth/*``).
"""

from __future__ import annotations

import contextlib

from astroquery.esa.euclid import Euclid

from euclid_polish.catalog.client import EuclidCatalog

_catalog: EuclidCatalog | None = None
_user: str | None = None


def login(user: str, password: str) -> EuclidCatalog:
    """Authenticate and remember the session. Raises ``EuclidAuthError`` on failure."""
    global _catalog, _user
    _catalog = EuclidCatalog(login=user, password=password)
    _user = user
    return _catalog


def logout() -> None:
    """Drop the local session and log out of the archive (best effort)."""
    global _catalog, _user
    _catalog = None
    _user = None
    with contextlib.suppress(Exception):
        Euclid.logout()


def is_authenticated() -> bool:
    return _catalog is not None


def current_user() -> str | None:
    return _user


def catalog() -> EuclidCatalog | None:
    return _catalog
