"""
Shared CLI utilities for EuclidPolish CLI.

This module provides common functionality used across multiple CLI commands,
including input validation and display formatting.
"""

from euclid_polish.config import Config


class ValidationResult:
    """Reusable validation functions for user input."""

    @staticmethod
    def validate_ra(value: str) -> bool | str:
        """
        Validate Right Ascension input.

        Parameters:
        -----------
        value : str
            User input string.

        Returns:
        --------
        bool or str
            True if valid, error message string if invalid.
        """
        if len(value) == 0:
            return 'RA is required'
        try:
            ra = float(value)
            if not (Config.RA_MIN <= ra < Config.RA_MAX):
                return f'RA must be between {Config.RA_MIN} and {Config.RA_MAX} degrees, got {ra}'
        except ValueError:
            return 'RA must be a number'
        return True

    @staticmethod
    def validate_dec(value: str) -> bool | str:
        """
        Validate Declination input.

        Parameters:
        -----------
        value : str
            User input string.

        Returns:
        --------
        bool or str
            True if valid, error message string if invalid.
        """
        if len(value) == 0:
            return 'Dec is required'
        try:
            dec = float(value)
            if not (Config.DEC_MIN <= dec <= Config.DEC_MAX):
                return f'Dec must be between {Config.DEC_MIN} and +{Config.DEC_MAX} degrees, got {dec}'
        except ValueError:
            return 'Dec must be a number'
        return True

    @staticmethod
    def validate_positive_number(value: str, field_name: str = "Value") -> bool | str:
        """
        Validate that input is a positive number.

        Parameters:
        -----------
        value : str
            User input string.
        field_name : str
            Name of the field for error messages.

        Returns:
        --------
        bool or str
            True if valid, error message string if invalid.
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


class DisplayFormatter:
    """Utility class for consistent display formatting."""

    @staticmethod
    def print_header(title: str) -> None:
        """
        Print a formatted header.

        Parameters:
        -----------
        title : str
            Header title.
        """
        print("\n" + "=" * Config.HEADER_WIDTH)
        print(f"  {title}")
        print("=" * Config.HEADER_WIDTH + "\n")

    @staticmethod
    def print_success(message: str) -> None:
        """Print a success message."""
        print(f"\n{Config.SUCCESS_PREFIX} {message}")

    @staticmethod
    def print_error(message: str) -> None:
        """Print an error message."""
        print(f"\n{Config.ERROR_PREFIX} {message}")

    @staticmethod
    def print_cancelled() -> None:
        """Print a cancelled message."""
        print(f"\n{Config.ERROR_PREFIX} Cancelled")


def build_command_args(arg_dict: dict) -> list:
    """
    Build command argument list from dictionary.

    Parameters:
    -----------
    arg_dict : dict
        Dictionary of argument names to values.
        Values of None are skipped.

    Returns:
    --------
    list
        Flat list of arguments ready for subprocess.
    """
    args = []
    for key, value in arg_dict.items():
        if value is not None:
            args.extend([f"--{key}", str(value)])
    return args
