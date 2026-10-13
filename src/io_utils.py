"""Reading and writing JSON files, validated with pydantic models."""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from src.models import FunctionDefinition, PromptInput


def read_json_file(path: str | Path) -> Any:
    """Read and parse a JSON file, with clear error messages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file, is empty, is not
            valid UTF-8, or is not valid JSON.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8: {file_path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read file: {file_path}") from exc
    if not content.strip():
        raise ValueError(f"File is empty: {file_path}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {file_path}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc


def write_json_file(path: str | Path, payload: Any) -> None:
    """Write a JSON-serializable payload to disk, creating parent dirs.

    Raises:
        ValueError: If the payload cannot be serialized to valid JSON.
        OSError: If the file cannot be written.
    """
    file_path = Path(path)
    try:
        serialized = json.dumps(
            payload, ensure_ascii=False, indent=2, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Payload is not valid JSON: {exc}") from exc
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"{serialized}\n", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to write JSON file: {file_path}") from exc


def _describe_error(error: ErrorDetails) -> str:
    """Turn one pydantic error into a short and precise sentence.

    Example: ``'name' must not be empty``.
    """
    kind = error["type"]
    location = error["loc"]
    if not location:
        if kind in ("model_type", "model_attributes_type", "dict_type"):
            return "entry must be a JSON object"
        return f"entry is invalid: {error['msg']}"
    field = ".".join(str(part) for part in location)
    context = error.get("ctx") or {}
    if kind == "missing":
        problem = "is missing"
    elif kind == "string_too_short":
        minimum = context.get("min_length", 1)
        if minimum == 1:
            problem = "must not be empty"
        else:
            problem = f"must have at least {minimum} characters"
    elif kind == "string_type":
        problem = "must be a string"
    elif kind == "dict_type":
        problem = "must be an object"
    elif kind == "literal_error":
        problem = (
            f"must be one of {context.get('expected')} "
            f"(got {error['input']!r})"
        )
    elif kind == "extra_forbidden":
        problem = "is not allowed here"
    elif kind == "value_error":
        problem = f"is invalid: {context.get('error', error['msg'])}"
    else:
        problem = f"is invalid: {error['msg']}"
    return f"'{field}' {problem}"


def _function_label(position: int, item: Any) -> str:
    """Return a label such as ``function #2 ('fn_add')`` for messages."""
    name = item.get("name") if isinstance(item, dict) else None
    if isinstance(name, str) and name.strip():
        return f"function #{position} ({name!r})"
    return f"function #{position}"


def load_function_definitions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate functions_definition.json.

    Every problem found is reported at once, with the position of the
    function (starting at 1) and the exact field concerned.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed, empty, or contains
            invalid or duplicate function definitions.
    """
    payload = read_json_file(path)
    if not isinstance(payload, list):
        raise ValueError(f"Function definitions must be a JSON array: {path}")
    if not payload:
        raise ValueError(f"No function defined in {path}")
    functions: list[FunctionDefinition] = []
    problems: list[str] = []
    for position, item in enumerate(payload, start=1):
        label = _function_label(position, item)
        try:
            function = FunctionDefinition.model_validate(item)
        except ValidationError as exc:
            problems.extend(
                f"{label}: {_describe_error(error)}"
                for error in exc.errors()
            )
            continue
        if not function.name.strip():
            problems.append(f"{label}: 'name' must not be blank")
            continue
        for parameter in function.parameters:
            if not parameter.strip():
                problems.append(
                    f"{label}: a parameter name must not be blank"
                )
        functions.append(function)
    names = [function.name for function in functions]
    for name in sorted({name for name in names if names.count(name) > 1}):
        problems.append(f"duplicate function name {name!r}")
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(f"Invalid function definitions in {path}:\n{details}")
    return functions


def load_prompts(path: str | Path) -> list[PromptInput]:
    """Load function_calling_tests.json, skipping invalid entries.

    Each invalid entry is reported on stderr with its position (starting
    at 1) and the reason, then ignored.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed or not a JSON array.
    """
    payload = read_json_file(path)
    if not isinstance(payload, list):
        raise ValueError(f"Prompts must be a JSON array: {path}")
    prompts: list[PromptInput] = []
    for position, item in enumerate(payload, start=1):
        try:
            prompt = PromptInput.model_validate(item)
        except ValidationError as exc:
            reasons = "; ".join(
                _describe_error(error) for error in exc.errors()
            )
            print(
                f"Error: prompt #{position} skipped: {reasons}",
                file=sys.stderr,
            )
            continue
        if not prompt.prompt.strip():
            print(
                f"Error: prompt #{position} skipped: "
                "'prompt' must not be empty",
                file=sys.stderr,
            )
            continue
        prompts.append(prompt)
    return prompts
