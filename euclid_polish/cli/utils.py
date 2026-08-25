"""
Shared CLI helpers for EuclidPolish.

Plain module-level functions for user-input validation and console display.
These were previously grouped under stateless ``ValidationResult`` /
``DisplayFormatter`` namespaces; since neither held any state, they are just
functions.
"""

from typing import Literal

from euclid_polish.config import Config

# ---------------------------------------------------------------------------
# Input validation — each returns ``True`` if valid, else an error-message str.
# ---------------------------------------------------------------------------

def validate_ra(value: str) -> Literal[True] | str:
    """Validate a Right Ascension string. ``True`` if valid, else an error message."""
    if len(value) == 0:
        return 'RA is required'
    try:
        ra = float(value)
        if not (Config.RA_MIN <= ra < Config.RA_MAX):
            return f'RA must be between {Config.RA_MIN} and {Config.RA_MAX} degrees, got {ra}'
    except ValueError:
        return 'RA must be a number'
    return True


def validate_dec(value: str) -> Literal[True] | str:
    """Validate a Declination string. ``True`` if valid, else an error message."""
    if len(value) == 0:
        return 'Dec is required'
    try:
        dec = float(value)
        if not (Config.DEC_MIN <= dec <= Config.DEC_MAX):
            return f'Dec must be between {Config.DEC_MIN} and +{Config.DEC_MAX} degrees, got {dec}'
    except ValueError:
        return 'Dec must be a number'
    return True


def validate_positive_number(value: str, field_name: str = "Value") -> Literal[True] | str:
    """Validate that ``value`` parses to a positive number.

    ``True`` if valid, else an error message naming ``field_name``.
    """
    if len(value) == 0:
        return f'{field_name} is required'
    try:
        num = float(value)
        if num <= 0:
            return f'{field_name} must be positive, got {num}'
    except ValueError:
        return f'{field_name} must be a number'
    return True


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * Config.HEADER_WIDTH)
    print(f"  {title}")
    print("=" * Config.HEADER_WIDTH + "\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"\n{Config.SUCCESS_PREFIX} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"\n{Config.ERROR_PREFIX} {message}")


def print_cancelled() -> None:
    """Print a cancelled message."""
    print(f"\n{Config.ERROR_PREFIX} Cancelled")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def build_command_args(arg_dict: dict) -> list:
    """Flatten ``{name: value}`` into ``["--name", "value", ...]`` (skips ``None``)."""
    args = []
    for key, value in arg_dict.items():
        if value is not None:
            args.extend([f"--{key}", str(value)])
    return args
