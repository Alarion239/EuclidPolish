"""The automatic stale-cube purge: the durable request flag, the idle-time
job the registry hook starts, and the end-to-end effect on the member-cube
buckets (an Evaluate after a purge infers only what the purge removed)."""
from __future__ import annotations

import threading
import time

import pytest

from euclid_polish.web.helpers import purge_requests


def test_requests_accumulate_and_clear_only_with_their_token():
    assert purge_requests.read_pending() is None
    purge_requests.request_stale_purge("archived member_03")
    first = purge_requests.read_pending()
    purge_requests.request_stale_purge("pulled member_04")
    second = purge_requests.read_pending()

    assert second["reasons"] == ["archived member_03", "pulled member_04"]
    assert purge_requests.clear_pending(first["requested_at"]) is False   # newer request
    assert purge_requests.clear_pending(second["requested_at"]) is True
    assert purge_requests.read_pending() is None
