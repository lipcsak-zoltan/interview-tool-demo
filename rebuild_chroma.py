#!/usr/bin/env python3
"""Compatibility wrapper for scripts/rebuild_chroma.py."""

from __future__ import annotations

from scripts.rebuild_chroma import main


if __name__ == "__main__":
    raise SystemExit(main())
