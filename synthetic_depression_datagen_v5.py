"""
synthetic_depression_datagen_v5.py
──────────────────────────────────
Generates synthetic doctor‑patient dialogues for depression‑screening research.
Uses Big Five personality templates to drive depression presentation patterns.

This is a thin wrapper that calls the modular synthetic_datagen package.
For the actual implementation, see synthetic_datagen/ directory.

Usage:
    python synthetic_depression_datagen_v5.py

Or use the CLI directly:
    python -m synthetic_datagen.cli --help
"""

if __name__ == "__main__":
    from synthetic_datagen.cli import main
    main()
