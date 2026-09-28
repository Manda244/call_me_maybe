"""Reading and writing JSON files, with schema validation via pydantic models."""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDefinition, PromptInput


def read_json_file(path: str | Path) -> Any:
    """Read and parse a JSON file, with clear error messages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file, is empty, or is not valid JSON.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to read file: {file_path}") from exc
    if not content.strip():
        raise ValueError(f"File is empty: {file_path}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {file_path}") from exc


def write_json_file(path: str | Path, payload: Any) -> None:
    """Write a JSON-serializable payload to disk, creating parent dirs.

    Raises:
        OSError: If the file cannot be written.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        file_path.write_text(f"{serialized}\n", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to write JSON file: {file_path}") from exc


def load_function_definitions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate functions_definition.json.

    Raises:
        ValueError: If the file is missing, malformed, or does not
            match the expected schema.
    """
    payload = read_json_file(path)
    if not isinstance(payload, list):
        raise ValueError(f"Function definitions must be a JSON array: {path}")
    try:
        return TypeAdapter(list[FunctionDefinition]).validate_python(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid function definitions in {path}: {exc}") from exc


def load_prompts(path: str | Path) -> list[PromptInput]:
    """Load and validate function_calling_tests.json.

    Raises:
        ValueError: If the file is missing, malformed, or does not
            match the expected schema.
    """
    payload = read_json_file(path)
    if not isinstance(payload, list):
        raise ValueError(f"Prompts must be a JSON array: {path}")
    try:
        return TypeAdapter(list[PromptInput]).validate_python(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid prompts in {path}: {exc}") from exc
