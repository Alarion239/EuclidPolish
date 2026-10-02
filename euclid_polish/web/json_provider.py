"""Standard JSON for every response of the console.

Python's ``json`` writes a non-finite float as a bare ``NaN`` / ``Infinity``
token, which is not JSON: the browser's ``JSON.parse`` rejects the WHOLE
response, so one NaN deep in a payload (an empty group's mean PSNR in a saved
compare report) made a page fail with "response … is not valid JSON".
:class:`StandardJSONProvider` sends every non-finite float as ``null``. The
strict encode is tried first, so a finite payload costs one ``json.dumps``.
"""

from __future__ import annotations

import json
import math
from typing import Any

from flask.json.provider import DefaultJSONProvider


def finite_or_none(value: Any) -> Any:
    """``value`` with every NaN/±Infinity float (inside dicts, lists and
    tuples) replaced by ``None``; other values are returned as they are."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    return value


class StandardJSONProvider(DefaultJSONProvider):
    """Flask's JSON provider, sending non-finite floats as ``null``."""

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        default = kwargs.pop("default", self.default)
        kwargs.setdefault("ensure_ascii", self.ensure_ascii)
        kwargs.setdefault("sort_keys", self.sort_keys)
        try:
            return json.dumps(obj, default=default, allow_nan=False, **kwargs)
        except ValueError:
            # ``default`` converts dataclasses, dates… whose fields may be NaN too.
            return json.dumps(finite_or_none(obj), allow_nan=False,
                              default=lambda o: finite_or_none(default(o)), **kwargs)


__all__ = ["StandardJSONProvider", "finite_or_none"]
