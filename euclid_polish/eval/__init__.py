"""Catalog-based evaluation helpers (metrics, morphology, image prep).

This package holds the *pure* (no TensorFlow / PyTorch) building blocks for the
evaluation pipeline so they can be unit-tested without heavyweight ML deps. The
model-driven entry points live in ``scripts/`` and import their frameworks
lazily.
"""
