"""Constrained generation of function calls with the small LLM."""

import sys
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from llm_sdk import Small_LLM_Model

from src.constraints import (
    MAX_NUMBER_DIGITS,
    JsonCallConstraint,
    NumberTooLargeError,
)
from src.io_utils import (
    load_function_definitions,
    load_prompts,
    write_json_file,
)
from src.models import FunctionCallResult, FunctionDefinition
from src.vocab import Vocabulary

MAX_NAME_TOKENS = 64
MAX_PARAMETER_TOKENS = 512
NAME_TERMINATOR = '"'


def build_token_table(vocabulary: Vocabulary) -> dict[int, str]:
    """Map every usable token id to its literal text.

    Empty tokens, special tokens (``<|...|>``) and tokens that are not
    valid UTF-8 on their own (fragments of a multi-byte character) are
    removed. The table is built once and reused for every prompt.
    """
    table: dict[int, str] = {}
    for token_id, data in vocabulary.token_bytes.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and not text.startswith("<|"):
            table[token_id] = text
    return table


def _encode(model: Small_LLM_Model, text: str) -> list[int]:
    """Encode text and normalize the SDK result to a flat list of ids."""
    rows = model.encode(text).tolist()
    if rows and isinstance(rows[0], list):
        rows = rows[0]
    if not isinstance(rows, list):
        raise TypeError("llm_sdk.encode() returned an unexpected shape")
    return [int(token_id) for token_id in rows]


def _ranked_ids(logits: list[float]) -> list[int]:
    """Return token ids sorted by decreasing logit."""
    order = np.argsort(np.asarray(logits, dtype=np.float64))[::-1]
    return [int(token_id) for token_id in order]


def _pick_token(
    logits: list[float],
    table: dict[int, str],
    is_allowed: Callable[[str], bool],
) -> tuple[int, str]:
    """Return the best-scoring token whose text satisfies the constraint.

    This is equivalent to setting the logits of forbidden tokens to
    ``-inf`` and taking the argmax, but it stops at the first allowed
    token instead of testing the whole vocabulary.

    Raises:
        ValueError: If no token satisfies the constraint.
    """
    for token_id in _ranked_ids(logits):
        text = table.get(token_id)
        if text and is_allowed(text):
            return token_id, text
    raise ValueError("No vocabulary token satisfies the constraint")


def _name_context(
    function_definitions: list[FunctionDefinition], prompt: str
) -> str:
    """Build the model context used to select a function name."""
    lines = []
    for function in function_definitions:
        arguments = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in function.parameters.items()
        )
        lines.append(f"- {function.name}({arguments}): {function.description}")
    return (
        "Available functions:\n"
        + "\n".join(lines)
        + "\n\nRequest: "
        + prompt
        + '\nName of the function to call: "'
    )


def _name_allows(names: list[str], generated: str, text: str) -> bool:
    """Return whether ``text`` may follow ``generated`` in a function name.

    The closing quote is only allowed once ``generated`` is a complete
    name, which lets the model choose between a name and a longer one
    that starts with it (``fn_add`` / ``fn_add_numbers``).
    """
    if generated in names and text == NAME_TERMINATOR:
        return True
    candidate = generated + text
    return any(name.startswith(candidate) for name in names)


def choose_function_name(
    model: Small_LLM_Model,
    table: dict[int, str],
    function_definitions: list[FunctionDefinition],
    prompt: str,
) -> str:
    """Let the LLM select one function name, token by token.

    Only tokens that keep the text a prefix of an existing function name
    are allowed, so the result is always a valid name.

    Raises:
        ValueError: If no name is produced within the token limit.
    """
    names = [function.name for function in function_definitions]
    input_ids = _encode(model, _name_context(function_definitions, prompt))
    generated = ""
    for _ in range(MAX_NAME_TOKENS):
        extendable = any(
            name != generated and name.startswith(generated)
            for name in names
        )
        if generated in names and not extendable:
            return generated
        logits = model.get_logits_from_input_ids(input_ids)
        token_id, text = _pick_token(
            logits, table, partial(_name_allows, names, generated)
        )
        if text == NAME_TERMINATOR:
            return generated
        generated += text
        input_ids.append(token_id)
    raise ValueError("Function name generation exceeded its token limit")


def _parameter_context(
    function_definition: FunctionDefinition, prompt: str
) -> str:
    """Build the model context used to generate the arguments."""
    arguments = "\n".join(
        f"- {name} ({spec.type})"
        for name, spec in function_definition.parameters.items()
    )
    return (
        "Extract the arguments of a call to the function "
        f"{function_definition.name}: {function_definition.description}\n"
        f"Arguments:\n{arguments}\n\n"
        f"Request: {prompt}\n"
        "Copy the values from the request. Do not answer the request.\n"
        "JSON call:"
    )


def generate_parameters(
    model: Small_LLM_Model,
    table: dict[int, str],
    function_definition: FunctionDefinition,
    prompt: str,
) -> dict[str, str | float | int | bool]:
    """Generate the arguments with schema-constrained decoding.

    Raises:
        NumberTooLargeError: If the model wants a number with more than
            ``MAX_NUMBER_DIGITS`` digits.
        ValueError: If the JSON is not complete within the token limit.
    """
    constraint = JsonCallConstraint(
        function_definition=function_definition,
        function_name=function_definition.name,
    )
    input_ids = _encode(
        model, _parameter_context(function_definition, prompt)
    )
    for _ in range(MAX_PARAMETER_TOKENS):
        if constraint.is_complete():
            break
        logits = model.get_logits_from_input_ids(input_ids)
        favorite = table.get(int(np.argmax(np.asarray(logits))), "")
        if favorite and constraint.would_overflow(favorite):
            raise NumberTooLargeError(
                f"Number exceeds {MAX_NUMBER_DIGITS} digits "
                f"(maximum {'9' * MAX_NUMBER_DIGITS})"
            )
        token_id, text = _pick_token(logits, table, constraint.is_valid_token)
        constraint.consume(text)
        input_ids.append(token_id)
    if not constraint.is_complete():
        raise ValueError("Parameter generation exceeded its token limit")
    return constraint.resolved_parameters()


def generate_result(
    model: Small_LLM_Model,
    table: dict[int, str],
    function_definitions: list[FunctionDefinition],
    prompt: str,
) -> dict[str, object]:
    """Produce one function-call result for one prompt."""
    name = choose_function_name(model, table, function_definitions, prompt)
    function_definition = next(
        function
        for function in function_definitions
        if function.name == name
    )
    parameters = generate_parameters(
        model, table, function_definition, prompt
    )
    result = FunctionCallResult(
        prompt=prompt, name=name, parameters=parameters
    )
    return result.model_dump()


def run_pipeline(
    functions_path: str | Path,
    tests_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, object]]:
    """Process every prompt and write the results file.

    A prompt that fails is reported on stderr and skipped.

    Raises:
        FileNotFoundError, ValueError, OSError: If an input file is
            missing or invalid, or the model cannot be loaded.
    """
    function_definitions = load_function_definitions(functions_path)
    prompts = load_prompts(tests_path)
    model = Small_LLM_Model()
    vocabulary = Vocabulary.from_file(model.get_path_to_vocab_file())
    table = build_token_table(vocabulary)
    results: list[dict[str, object]] = []
    for prompt_item in prompts:
        try:
            results.append(
                generate_result(
                    model, table, function_definitions, prompt_item.prompt
                )
            )
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError) \
                as exc:
            print(
                f"Error: prompt {prompt_item.prompt!r} skipped: {exc}",
                file=sys.stderr,
            )
    write_json_file(output_path, results)
    return results
