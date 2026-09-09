import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .model import FunctionsDefinitions


def load_fonction_definitions(
    read_file: str | Path | None = None,
) -> list[FunctionsDefinitions]:
    """Load the function definitions from the JSON file."""
    if read_file is None:
        read_file = Path(__file__).resolve().parent.parent / "data" / "input" / "functions_definition.json"
    adapter = TypeAdapter(list[FunctionsDefinitions])
    with open(read_file, encoding="utf-8") as file:
        return adapter.validate_python(json.load(file))


def load_test_prompts(
    read_file: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load the test prompts from the JSON file."""
    if read_file is None:
        read_file = Path(__file__).resolve().parent.parent / "data" / "input" / "function_calling_tests.json"
    adapter = TypeAdapter(list[dict[str, str]])
    with open(read_file, encoding="utf-8") as file:
        return adapter.validate_python(json.load(file))


def write_function_calling_results(
    results: list[dict[str, Any]],
    write_file: str | Path | None = None,
) -> None:
    """Write the function calling results to a JSON file."""
    if write_file is None:
        write_file = Path(__file__).resolve().parent.parent / "data" / "output" / "function_calling_results.json"
    output_path = Path(write_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
