#!/usr/bin/env python3
"""
EuclidPolish - Super-resolution for astronomical images.

Main entry point for the EuclidPolish package.

Usage:
    python main.py euclid
    python main.py sky
    python main.py training
    python main.py visualization
"""

from euclid_polish.cli.main import main

if __name__ == "__main__":
    main()
