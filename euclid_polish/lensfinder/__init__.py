"""Lens-finder: measure the super-resolution lens-discovery gain.

Replicates the POLISH++ evaluation (Wu, Connor, McCarty & Bouman 2026,
arXiv:2603.09162): finetune a Zoobot binary lens classifier and compare its
recovery on LR vs SR vs HR source-centered stamps, reporting TPR-vs-Einstein-
radius at a fixed false-positive rate.

The package is split so the TensorFlow side (stamp building, SR inference) and
the PyTorch/Zoobot side (training) never import each other; the cross-env
contract is a catalog CSV of PNG file paths. ``catalog``, ``metrics`` and
``predictions`` are pure (numpy/csv only) and unit-testable in the main env;
``stamps`` defers astropy/PIL imports into functions.
"""
