"""
EuclidPolish - Super-resolution for astronomical images.

This package provides tools for:
- Euclid telescope data download and PSF extraction
- Clean and dirty sky generation
- WDSR super-resolution model training
- Visualization utilities

Example:
    >>> from euclid_polish.config import Config
    >>> from euclid_polish.euclid import StarCatalog
    >>> from euclid_polish.training import Trainer
"""

__version__ = "0.1.0"

# Core configuration
from euclid_polish.config import Config

__all__ = ["Config"]
