"""Command-line entry point for the function-calling pipeline."""

import argparse
import sys
from pathlib import Path

from src.generator import run_pipeline

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, falling back to default paths."""
    parser = argparse.ArgumentParser(
        description="Run function-calling generation with constrained decoding."
    )
    parser.add_argument("--functions_definition", type=Path, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run the pipeline and report any error without crashing."""
    args = parse_args()
    try:
        run_pipeline(args.functions_definition, args.input, args.output)
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
