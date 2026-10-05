"""Evaluation of the SR ensemble: catalog/grouped runners, metrics, combiners.

Many modules here (e.g. :mod:`.combiner`, :mod:`.spatial_gate`,
:mod:`.ensemble_diagnostics`, the catalog readers) are *pure* (NumPy, no
TensorFlow) building blocks so they can be unit-tested without heavyweight ML
deps. TensorFlow is imported at module level by :mod:`.spatial_gate_fit` and,
through :mod:`euclid_polish.ensemble`, the TFRecord I/O or the training package,
by the model-driven runners (:mod:`.catalog_runner`, :mod:`.grouped_runner`,
:mod:`.synthetic_runner`, :mod:`.ensemble_infer`) and helpers such as
:mod:`.disagreement`, :mod:`.subsets` and :mod:`.power_spectrum`. The
``scripts/eval_*.py`` and ``scripts/fetch_*_catalog.py`` entry points are thin
CLI wrappers over these modules.
"""
