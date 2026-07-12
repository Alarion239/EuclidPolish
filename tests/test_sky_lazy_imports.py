from __future__ import annotations

import json
import subprocess
import sys


def test_observation_only_import_does_not_load_lens_generation():
    code = """
import json
import sys
import euclid_polish.sky.observation.observation_simulator
blocked = [name for name in sys.modules if (
    name == 'lenstronomy'
    or name.startswith('lenstronomy.')
    or name == 'euclid_polish.sky.generation.lens_population'
)]
print(json.dumps(blocked))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_legacy_sky_reexports_resolve_lazily():
    import euclid_polish.sky as sky
    import euclid_polish.sky.generation as generation

    assert sky.ObservationSimulator.__module__.endswith("observation_simulator")
    assert sky.LensPopulation.__module__.endswith("lens_population")
    assert generation.SourceCatalogWriter.__module__.endswith("source_catalog")
    assert generation.SkySimulator.__module__.endswith("sky_simulator")
