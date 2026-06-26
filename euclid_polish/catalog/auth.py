"""
Euclid archive authentication.

Thin wrapper over ``astroquery.esa.euclid.Euclid.login/logout``.
All login/logout calls must go through this module so the local
``_is_logged_in`` flag stays coherent.
"""

from __future__ import annotations

import getpass
import os
import stat
import threading
from typing import Optional

from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.catalog.validator import validate_file_exists


_is_logged_in: bool = False
_login_user: Optional[str] = None
#: Serialise logins so concurrent re-auth (parallel band downloads refreshing an
#: expired TAP session) don't race on the astroquery session.
_login_lock = threading.Lock()


def is_authenticated() -> bool:
    """Return True if we believe we are logged into the Euclid archive."""
    return _is_logged_in


def current_user() -> Optional[str]:
    """Return the logged-in username if we logged in through this module."""
    return _login_user


def login_with_credentials(user: str, password: str) -> bool:
    """Log in with an explicit username/password."""
    global _is_logged_in, _login_user
    try:
        Euclid.login(user=user, password=password)
    except Exception as e:
        print(f"✗ Login failed: {e}")
        return False
    _is_logged_in = True
    _login_user = user
    return True


def login_with_file(credentials_file: str) -> bool:
    """
    Log in using a two-line credentials file.

    File format expected by astroquery: username on line 1, password on
    line 2. This is NOT ``.netrc``.
    """
    global _is_logged_in, _login_user
    path = os.path.expanduser(credentials_file)
    ok, err = validate_file_exists(path, "Credentials file")
    if not ok:
        print(f"✗ {err}")
        return False

    try:
        mode = os.stat(path).st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"⚠️  Warning: {path} is readable by group/others. Run: chmod 600 {path}")
    except OSError:
        pass

    try:
        Euclid.login(credentials_file=path)
    except Exception as e:
        print(f"✗ Login failed: {e}")
        return False

    user = None
    try:
        with open(path) as f:
            user = f.readline().strip() or None
    except OSError:
        pass

    _is_logged_in = True
    _login_user = user
    return True


def login_from_env() -> bool:
    """Log in using ``EUCLID_USER`` / ``EUCLID_PASSWORD``. False if unset."""
    user = os.environ.get("EUCLID_USER")
    password = os.environ.get("EUCLID_PASSWORD")
    if not user or not password:
        return False
    return login_with_credentials(user, password)


def login_interactive() -> bool:
    """Prompt for credentials at the terminal (password via ``getpass``)."""
    try:
        user = input("Euclid username: ").strip()
        if not user:
            print("✗ Empty username")
            return False
        password = getpass.getpass("Euclid password: ")
    except (KeyboardInterrupt, EOFError):
        print("\n✗ Login cancelled")
        return False
    if not password:
        print("✗ Empty password")
        return False
    return login_with_credentials(user, password)


def login(credentials_file: Optional[str] = None, allow_interactive: bool = True) -> bool:
    """
    High-level login: env vars → file → interactive.

    Precedence:
      1. ``login_from_env()`` (``EUCLID_USER``/``EUCLID_PASSWORD``)
      2. ``login_with_file(credentials_file or Config.DEFAULT_CREDENTIALS_FILE)``
         if such a file exists on disk
      3. ``login_interactive()`` if ``allow_interactive``

    Serialised by ``_login_lock`` so concurrent re-authentication (e.g. parallel
    band downloads each refreshing an expired TAP session) doesn't corrupt the
    shared astroquery session.
    """
    with _login_lock:
        if login_from_env():
            return True

        path = credentials_file or Config.DEFAULT_CREDENTIALS_FILE
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            if login_with_file(expanded):
                return True

        if allow_interactive:
            return login_interactive()
        return False


def logout() -> bool:
    """Log out from the Euclid archive; always clears local state."""
    global _is_logged_in, _login_user
    ok = True
    try:
        Euclid.logout()
    except Exception as e:
        print(f"⚠️  Logout raised (clearing local state anyway): {e}")
        ok = False
    _is_logged_in = False
    _login_user = None
    return ok


def check_connection() -> bool:
    """Liveness probe against the Euclid TAP endpoint."""
    try:
        Euclid.get_status_messages()
        return True
    except Exception:
        return False
