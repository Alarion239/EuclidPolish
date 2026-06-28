"""Laptop-side Euclid archive session for the web UI.

Holds the :class:`~euclid_polish.catalog.client.EuclidCatalog` the login form
created, so the catalog page can show "logged in as <user>". The archive
download itself runs on FASRC; this is only the local UI's session state.
"""

from __future__ import annotations

from typing import Optional

from astroquery.esa.euclid import Euclid

from euclid_polish.catalog.client import EuclidCatalog

_catalog: Optional[EuclidCatalog] = None
_user: Optional[str] = None


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
    try:
        Euclid.logout()
    except Exception:
        pass


def is_authenticated() -> bool:
    return _catalog is not None


def current_user() -> Optional[str]:
    return _user


def catalog() -> Optional[EuclidCatalog]:
    return _catalog
