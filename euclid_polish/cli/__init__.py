"""
CLI framework for EuclidPolish.

This module provides a unified command-line interface for all EuclidPolish operations.
"""

from euclid_polish.cli.utils import (
    build_command_args,
    print_cancelled,
    print_error,
    print_header,
    print_success,
    validate_dec,
    validate_positive_number,
    validate_ra,
)

__all__ = [
    "build_command_args",
    "print_cancelled",
    "print_error",
    "print_header",
    "print_success",
    "validate_dec",
    "validate_positive_number",
    "validate_ra",
]
