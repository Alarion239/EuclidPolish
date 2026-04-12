"""
Base class for CLI commands.

This module provides the abstract base class for all CLI commands.
"""

from abc import ABC, abstractmethod
from typing import Optional


class CLICommand(ABC):
    """
    Abstract base class for CLI commands.

    All CLI commands must inherit from this class and implement
    the required methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the command name (used for routing)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the command."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> bool:
        """
        Execute the command.

        Parameters:
        -----------
        **kwargs : dict
            Command-specific arguments.

        Returns:
        --------
        bool
            True if command succeeded, False otherwise.
        """
        pass

    def validate_args(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate command arguments (optional override).

        Parameters:
        -----------
        **kwargs : dict
            Command arguments to validate.

        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        return True, None
