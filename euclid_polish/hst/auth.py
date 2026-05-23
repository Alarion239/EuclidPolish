"""
HST archive authentication shim.

Public HST data products (drizzled science images from completed
non-proprietary programs) are open access and need no login. This module
exists for symmetry with :mod:`euclid_polish.euclid.auth` so the rest of
the codebase can ``from euclid_polish.hst import auth`` without branching.

If a future feature ever needs MAST credentials (proprietary observations,
quota-protected services), wire ``astroquery.mast.Observations.login`` in
here following the same pattern as the Euclid module.
"""

from __future__ import annotations


def is_authenticated() -> bool:
    """Always ``True``: public HST data needs no authentication."""
    return True


def current_user() -> str:
    """Public-data sentinel — no real user is logged in."""
    return "<public-MAST>"


def login(*_args, **_kwargs) -> bool:
    """No-op: public HST data needs no login. Returns ``True``."""
    return True


def logout() -> bool:
    """No-op: nothing to log out from."""
    return True
