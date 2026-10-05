#!/usr/bin/env python3
"""
EuclidPolish - Super-resolution for astronomical images.

Main entry point for the EuclidPolish package.

Usage:
    python main.py

Starts the interactive menu (Euclid operations, sky generation, model
training, visualization); command-line arguments are ignored.
"""

from euclid_polish.cli.main import main

if __name__ == "__main__":
    main()
