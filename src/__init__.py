"""Public API for the function-calling pipeline.

This package exposes the main pipeline entry point used by the CLI and
higher-level integrations.
"""

from .generator import run_pipeline

__all__ = ["run_pipeline"]
