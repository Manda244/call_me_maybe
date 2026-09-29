"""Command-line entry point for the function-calling pipeline."""

import argparse
import sys
from pathlib import Path

from src.generator import run_pipeline

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, falling back to default paths.

    Args:
        argv: Arguments to parse; ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural-language prompts into function calls "
            "using constrained decoding."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help=f"functions definition file (default: {DEFAULT_FUNCTIONS})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"prompts file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"results file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and report any error without a traceback.

    Returns:
        0 on success, 1 on error, 130 if interrupted by the user.
    """
    args = parse_args(argv)
    try:
        results = run_pipeline(
            args.functions_definition, args.input, args.output
        )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError) as exc:
        # OSError also covers FileNotFoundError.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - last resort, never crash
        print(
            f"Unexpected error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
