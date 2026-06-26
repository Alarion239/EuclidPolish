"""The image package must stay at the bottom of the import graph.

`Image`/`ImageSet` own only self-contained operations; they must never import an
operator (simulator, forward model, trained model, archive downloader) or any
heavy domain subsystem. This guards the layering invariant.
"""
import ast
import os

import pytest

_PKG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "euclid_polish", "image")

# Substrings that, if found in an absolute import inside image/, mean a layering
# violation (image reaching "up" into operators/CLI/web/eval/training).
_FORBIDDEN = (
    "euclid_polish.sky.multiband_forward",
    "euclid_polish.sky.multiband_generator",
    "euclid_polish.model",
    "euclid_polish.training",
    "euclid_polish.eval",
    "euclid_polish.cli",
    "euclid_polish.web",
    "euclid_polish.cutout",
    "euclid_polish.euclid",
)


def _modules():
    for fn in os.listdir(_PKG):
        if fn.endswith(".py"):
            yield os.path.join(_PKG, fn)


@pytest.mark.parametrize("path", list(_modules()))
def test_no_upward_imports(path):
    tree = ast.parse(open(path).read(), filename=path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    for mod in names:
        for bad in _FORBIDDEN:
            assert not mod.startswith(bad), (
                f"{os.path.basename(path)} imports {mod}: image/ must not depend "
                f"on operators/CLI/web/eval/training")
