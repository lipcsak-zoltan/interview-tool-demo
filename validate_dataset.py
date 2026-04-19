#!/usr/bin/env python3
"""Compatibility wrapper for scripts/validate_dataset.py."""

from __future__ import annotations

from scripts.validate_dataset import DEFAULT_DATASET_PATH, load_rows, main, validate_rows


__all__ = ["DEFAULT_DATASET_PATH", "load_rows", "main", "validate_rows"]


if __name__ == "__main__":
    raise SystemExit(main())
