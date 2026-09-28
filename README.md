*This project has been created as part of the 42 curriculum by marasolo.*

# Function Calling with Constrained Generation

## Description

This project builds a lightweight function-calling pipeline that reads JSON function definitions, selects a matching callable from a prompt, validates the result, and writes a valid output JSON payload.

## Instructions

1. Install dependencies with uv.
2. Run the application with the project default input files.
3. Provide custom input and output paths with CLI arguments.
4. Run the test suite and lint checks before finalizing changes.

## Resources

- Python 3.10+
- Pydantic models for validation
- Local LLM SDK based on Qwen/Qwen3-0.6B
- JSON-based function definitions and test prompts

## Algorithm explanation

The program reads a list of function definitions and prompts, selects the most relevant function name using the model, converts the prompt into parameter values, validates the result against the schema, and serializes the final output to JSON.

## Design decisions

- Keep the code modular across models, IO, vocabulary loading, and generation.
- Validate all user-facing and model-generated structures with Pydantic.
- Keep file operations isolated from generation logic.
- Use a single vocabulary object for token-to-ID and ID-to-token conversion.

## Performance analysis

The implementation minimizes repeated work by loading the vocabulary once per generation run and limiting the generation loop to a bounded token count. This keeps runtime manageable while still allowing the model to produce a valid result for each prompt.

## Challenges faced

- Matching the actual vocabulary format returned by the local SDK.
- Handling space representation in tokenizer vocabularies.
- Validating function calls with strict schema checks.
- Preventing invalid or incomplete JSON output.

## Testing strategy

The project includes a pytest suite covering JSON parsing, model validation, vocabulary handling, and CLI usage. The tests are designed to catch invalid JSON, missing files, invalid schemas, and incorrect result structures.

## Example usage

uv sync
uv run python -m src
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

## Constrained decoding

The project includes constrained generation logic that limits token choices based on the expected output structure and vocabulary. This reduces the risk of generating invalid JSON or unknown function names.

## LLM usage

The default runtime model is compatible with Qwen/Qwen3-0.6B via the local SDK bundled in the project. The function selection is driven by the model’s token probabilities rather than a hand-written mapping.

## Token handling

Token IDs are loaded from the vocabulary file and converted both ways to support selection, validation, and reconstruction of output strings. This is especially important for tokenizer-specific tokens such as spaces and special characters.
