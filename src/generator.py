import sys
from pathlib import Path
from typing import Protocol, cast

from llm_sdk import Small_LLM_Model

from src.constraints import JsonCallConstraint
from src.io_utils import load_function_definitions, load_prompts, write_json_file
from src.models import FunctionCallResult, FunctionDefinition, PromptInput
from src.vocab import Vocabulary


class ModelProtocol(Protocol):
    def encode(self, text: str) -> "EncodedIds":
        """Encode a prompt into model input ids."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits for the next token."""


class EncodedIds(Protocol):
    def tolist(self) -> list[list[int]]:
        """Return encoded ids as nested lists."""


def _literal_vocabulary(vocabulary: Vocabulary) -> dict[str, int]:
    """Convert raw vocabulary tokens to literal text and remove specials."""
    literal_vocabulary: dict[str, int] = {}
    for token_id, raw_token in vocabulary.id_to_token.items():
        token_text = vocabulary.token_to_text(raw_token)
        if not token_text or token_text.startswith("<|"):
            continue
        if token_text not in literal_vocabulary:
            literal_vocabulary[token_text] = token_id
    return literal_vocabulary


def _select_token(logits: list[float], allowed_ids: list[int]) -> int:
    """Mask forbidden logits and return the highest-scoring allowed id."""
    masked_logits = [float("-inf")] * len(logits)
    for token_id in allowed_ids:
        if 0 <= token_id < len(logits):
            masked_logits[token_id] = logits[token_id]
    if not allowed_ids or max(masked_logits) == float("-inf"):
        raise ValueError("No allowed token has a model logit")
    return max(range(len(masked_logits)), key=masked_logits.__getitem__)


def _encoded_ids(model: ModelProtocol, text: str) -> list[int]:
    """Encode text and normalize the SDK tensor-like result to integer ids."""
    batches = model.encode(text).tolist()
    if not isinstance(batches, list) or not batches or not isinstance(batches[0], list):
        raise TypeError("llm_sdk.encode() returned an unexpected shape")
    return [int(token_id) for token_id in batches[0]]


def choose_function_name(
    model: Small_LLM_Model,
    vocabulary: Vocabulary,
    function_definitions: list[FunctionDefinition],
    prompt: str,
) -> str:
    """Use constrained LLM generation to select one available function name."""
    names = [function.name for function in function_definitions]
    descriptions = "\n".join(
        function.name + ": " + function.description
        for function in function_definitions
    )
    context = (
        "Choose one function for this request. Return only its exact name.\n"
        + descriptions
        + "\nRequest: "
        + prompt
        + "\nFunction:"
    )
    input_ids = _encoded_ids(model, context)
    literal_vocabulary = _literal_vocabulary(vocabulary)
    generated_name = ""
    for _ in range(64):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed = {
            token_id: token_text
            for token_text, token_id in literal_vocabulary.items()
            if any(name.startswith(generated_name + token_text) for name in names)
        }
        selected = _select_token(logits, list(allowed))
        token_text = allowed[selected]
        generated_name += token_text
        input_ids.append(selected)
        if generated_name in names:
            return generated_name
    raise ValueError("Function name generation exceeded its token limit")


def _parameter_context(function_definition: FunctionDefinition, prompt: str) -> str:
    """Build the model context for schema-constrained parameter generation."""
    schema = ", ".join(
        parameter_name + ": " + parameter_type.type
        for parameter_name, parameter_type in function_definition.parameters.items()
    )
    return (
        "Return only the JSON object with the selected function name and parameters. "
        "Do not answer the request.\nFunction: "
        + function_definition.name
        + "\nParameters: "
        + schema
        + "\nRequest: "
        + prompt
        + "\nJSON:"
    )


def generate_parameters(
    model: Small_LLM_Model,
    vocabulary: Vocabulary,
    function_definition: FunctionDefinition,
    prompt: str,
) -> dict[str, str | float | int | bool]:
    """Generate and validate parameters with schema-constrained decoding."""
    constraint = JsonCallConstraint(
        function_definition=function_definition,
        function_name=function_definition.name,
    )
    input_ids = _encoded_ids(model, _parameter_context(function_definition, prompt))
    literal_vocabulary = _literal_vocabulary(vocabulary)
    token_limit = 512
    for _ in range(token_limit):
        allowed = constraint.get_allowed_tokens(literal_vocabulary)
        logits = model.get_logits_from_input_ids(input_ids)
        selected = _select_token(logits, [token_id for token_id, _ in allowed])
        token_text = next(text for token_id, text in allowed if token_id == selected)
        constraint.consume(token_text)
        input_ids.append(selected)
        if constraint.is_complete():
            break
    if not constraint.is_complete():
        raise ValueError("Parameter generation exceeded its token limit")
    return _validate_parameters(function_definition, constraint.resolved_parameters())


def _validate_parameters(
    function_definition: FunctionDefinition,
    parameters: dict[str, object],
) -> dict[str, str | float | int | bool]:
    """Validate generated names and values against the function schema."""
    expected = function_definition.parameters
    if set(parameters) != set(expected):
        raise ValueError("Generated parameter names do not match the schema")
    validated: dict[str, str | float | int | bool] = {}
    for parameter_name, parameter_type in expected.items():
        value = parameters[parameter_name]
        if parameter_type.type == "string" and type(value) is str:
            validated[parameter_name] = value
        elif parameter_type.type == "boolean" and type(value) is bool:
            validated[parameter_name] = value
        elif parameter_type.type == "integer" and type(value) is int:
            validated[parameter_name] = cast(int, value)
        elif parameter_type.type == "number" and type(value) in {int, float}:
            validated[parameter_name] = cast(int | float, value)
        else:
            raise ValueError("Generated parameter has an invalid type")
    return validated


def generate_result(
    model: Small_LLM_Model,
    vocabulary: Vocabulary,
    function_definitions: list[FunctionDefinition],
    prompt: str,
) -> dict[str, object]:
    """Produce one validated function-call result."""
    function_name = choose_function_name(model, vocabulary, function_definitions, prompt)
    function_definition = next(
        function for function in function_definitions if function.name == function_name
    )
    parameters = generate_parameters(model, vocabulary, function_definition, prompt)
    result = FunctionCallResult(
        prompt=prompt, name=function_name, parameters=parameters
    )
    return cast(dict[str, object], result.model_dump())


def run_pipeline(
    functions_path: str | Path, tests_path: str | Path, output_path: str | Path
) -> list[dict[str, object]]:
    """Run constrained generation for every input prompt and write the results."""
    function_definitions = load_function_definitions(functions_path)
    prompts: list[PromptInput] = load_prompts(tests_path)
    model = Small_LLM_Model()
    vocabulary = Vocabulary(model.get_path_to_vocab_file())
    results: list[dict[str, object]] = []
    for prompt_item in prompts:
        try:
            results.append(
                generate_result(
                    model, vocabulary, function_definitions, prompt_item.prompt
                )
            )
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            message = "Warning: skipped prompt " + repr(prompt_item.prompt)
            print(message + ": " + str(exc), file=sys.stderr)
    write_json_file(output_path, results)
    return results
