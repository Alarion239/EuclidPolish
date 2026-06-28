"""Default progress reporting for local eval runs.

The catalog / grouped / synthetic runners report progress through a plain
``on_progress(done, total, label)`` callback so the WebUI background job can
drive its own progress bar. When that callback is omitted — i.e. the loop is
run locally from a CLI or a notebook — there was previously *no* visible
progress at all, only log lines, which reads as "nothing is happening".

:func:`tqdm_progress` returns a ready-made callback backed by a lazily-created
``tqdm`` bar so any local run shows a real progress bar. The runners install it
as the default whenever the caller passes no ``on_progress``.
"""

from __future__ import annotations

from collections.abc import Callable

from tqdm.auto import tqdm


def tqdm_progress(desc: str = "eval") -> Callable[[int, int, str], None]:
    """Return an ``(done, total, label)`` callback backed by a ``tqdm`` bar.

    The bar is created on the first call (so ``total`` is known) and updated in
    place; the current label is shown as the bar's postfix. If ``total`` changes
    between phases the bar is resized. The bar closes itself once ``done`` reaches
    ``total`` so the caller's final ``(total, total, "done")`` tick tidies up.
    """
    state: dict = {"bar": None}

    def _cb(done: int, total: int, label: str = "") -> None:
        total = max(int(total), 1)
        bar = state["bar"]
        if bar is None:
            bar = state["bar"] = tqdm(total=total, desc=desc)
        if total != bar.total:
            bar.total = total
            bar.refresh()
        bar.n = min(int(done), total)
        if label:
            bar.set_postfix_str(label)
        bar.refresh()
        if bar.n >= total:
            bar.close()
            state["bar"] = None

    return _cb
