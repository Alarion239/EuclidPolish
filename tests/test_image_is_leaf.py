"""The image package must stay at the bottom of the import graph.

`Image`/`ImageSet` own only self-contained operations; they must never import an
operator (simulator, forward model, trained model, archive downloader) or any
heavy domain subsystem. This guards the layering invariant.
"""
import ast
import os

import pytest

import euclid_polish.image as image_package

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_REPO, "euclid_polish", "image")

# Module-path prefixes that, if an absolute import inside image/ starts with one,
# mean a layering violation (image reaching "up" into operators/CLI/web/eval/
# training). Each must name a real module or package, or it can never fire.
_FORBIDDEN = (
    "euclid_polish.sky.observation.observation_simulator",
    "euclid_polish.sky.generation.sky_simulator",
    "euclid_polish.model",
    # The ensemble is the trained model (also covers ensemble_registry).
    "euclid_polish.ensemble",
    "euclid_polish.training",
    "euclid_polish.eval",
    "euclid_polish.cli",
    "euclid_polish.web",
    # Catalog client + archive downloader.
    "euclid_polish.catalog",
    # visualization builds ON the image layer, never the reverse.
    "euclid_polish.visualization",
)


def _modules():
    for fn in os.listdir(_PKG):
        if fn.endswith(".py"):
            yield os.path.join(_PKG, fn)


def _imported_modules(source: str, filename: str = "<string>") -> list[str]:
    """Every absolute module name imported by ``source``."""
    tree = ast.parse(source, filename=filename)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    return names


def _violations(source: str, filename: str = "<string>") -> list[str]:
    """Imported modules in ``source`` that fall under a ``_FORBIDDEN`` prefix."""
    return [mod for mod in _imported_modules(source, filename)
            if any(mod.startswith(bad) for bad in _FORBIDDEN)]


@pytest.mark.parametrize("path", list(_modules()))
def test_no_upward_imports(path):
    bad = _violations(open(path).read(), filename=path)
    assert not bad, (
        f"{os.path.basename(path)} imports {bad}: image/ must not depend "
        f"on operators/CLI/web/eval/training")


@pytest.mark.parametrize("prefix", _FORBIDDEN)
def test_forbidden_prefixes_name_real_modules(prefix):
    """A prefix naming a module that does not exist can never fire."""
    base = os.path.join(_REPO, *prefix.split("."))
    assert os.path.isfile(base + ".py") or os.path.isfile(os.path.join(base, "__init__.py")), (
        f"_FORBIDDEN entry {prefix!r} matches no module under {_REPO}")


@pytest.mark.parametrize("module", [
    "euclid_polish.sky.generation.sky_simulator",
    "euclid_polish.sky.observation.observation_simulator",
    "euclid_polish.model",
    "euclid_polish.ensemble",
    "euclid_polish.catalog.downloader",
])
def test_guard_rejects_operator_imports(module):
    """The guard fires for each operator the module docstring names."""
    assert _violations(f"from {module} import X") == [module]
    assert _violations(f"import {module}") == [module]


def test_image_core_does_not_import_tensorflow():
    path = os.path.join(_PKG, "core.py")
    modules = _imported_modules(open(path).read(), filename=path)
    assert not any(module == "tensorflow" or module.startswith("tensorflow.")
                   for module in modules)


def test_image_package_does_not_export_cube_types():
    for name in ("AngularGrid", "CubeLike", "ImageCube", "PhysicalGrid", "PixelUnit"):
        assert name not in image_package.__all__
        assert not hasattr(image_package, name)
