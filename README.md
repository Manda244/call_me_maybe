*This project has been created as part of the 42 curriculum by marasolo.*

# Function Calling in LLMs

## Description

This project implements a compact function-calling pipeline for a local large language model. The goal is to transform a natural-language request into a valid function call whose name and arguments respect a predefined JSON schema.

Instead of letting the model generate raw text and hoping it matches the expected structure, the application performs constrained decoding. It selects a valid function name from a known list and then generates parameters token by token while checking that each step remains a valid prefix of the final JSON object. The result is a structured function call that is easier to validate, safer to execute, and much more reliable than unconstrained free-form generation.

The project is built around a small local LLM SDK and a set of JSON definitions describing callable functions and their parameters. It reads inputs from files, runs the generation pipeline, and writes the results to JSON output.

### Core features

- Parses a list of available functions and their schemas
- Selects the most relevant function from the user prompt
- Generates arguments under schema and type constraints
- Rejects invalid or incomplete JSON prefixes early
- Writes final outputs as executable function-call objects

## Instructions

### Prerequisites

- Python 3.10 or newer
- uv installed and available on your PATH
- A working local environment for the bundled LLM SDK

### Installation

From the project root:

```bash
uv sync
```

This installs the project dependencies and local workspace package configuration.

### Running the program

Run the default pipeline using the bundled example files:

```bash
uv run python -m src
```

This will read the default definitions and prompt inputs and produce the output file:

```text
data/output/function_calling_results.json
```

You can also specify a custom functions file, prompts file, and output path:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Validation

The project is designed to be checked with standard Python tooling and tests:

```bash
uv run pytest
```

If you are working on code quality as well, you can run additional checks supported by the project configuration.

## Algorithm explanation

The constrained decoding approach is the heart of the project.

1. Function selection
   - The model is given a list of available functions with names, parameter types, and short descriptions.
   - A prompt is built containing those definitions and the user request.
   - The model generates a function name token by token, but only tokens that keep the partial text as a valid prefix of an existing function name are accepted.
   - This avoids invalid names and allows names that share common prefixes such as `fn_add` and `fn_add_numbers`.

2. Parameter generation
   - Once the function is chosen, a second prompt asks the model to extract the arguments from the request.
   - The output is expected to be JSON, shaped as:

```json
{"name":"function_name","parameters":{"param1":value1,"param2":value2}}
```

   - A dedicated constraint object continuously checks whether the current partial text is still a valid prefix of this structure.

3. Token-level validation
   - Every candidate token is accepted only if appending it keeps the JSON valid.
   - String values must remain valid JSON strings, with escape sequences and quotes handled correctly.
   - Numbers must stay within a safe digit limit and cannot drift into invalid numeric forms.
   - Booleans are constrained to `true` or `false`.

4. Final validation and serialization
   - When the complete object is produced, the generated parameter dictionary is resolved and returned as a typed result.
   - The final output is written to JSON in a deterministic, machine-readable format.

This method is significantly safer than unconstrained generation because invalid branches are filtered before they can be emitted.

## Design decisions

Several implementation decisions were made to balance correctness, readability, and performance:

- Modular architecture: generation logic, JSON constraints, IO helpers, models, and vocabulary management are separated into different modules.
- Pydantic validation: function definitions and returned results are validated with typed models to prevent malformed data.
- Prefix-based decoding: generation is not based on raw completion; it is driven by a valid-prefix rule for the expected JSON output.
- Conservative numeric handling: numeric generation is limited to a safe number of digits to avoid overflow, invalid floats, and pathological edge cases.
- Vocabulary reuse: the token table is precomputed once and reused across prompts to avoid repeated decoding work.
- Fail-fast behavior: generation errors are reported clearly instead of silently producing broken output.

## Performance analysis

This solution prioritizes correctness and reliability over maximum raw throughput.

### Accuracy

The constrained decoder is very effective at preserving schema validity because it rejects tokens that would make the output impossible to complete. This substantially improves the chance that the generated result is both valid JSON and compatible with the selected function definition.

### Speed

The pipeline remains lightweight because it only processes a small number of tokens per function call, and the vocabulary is loaded once per run. The local model is intentionally compact, which keeps inference fast enough for experimentation and small benchmark sets.

### Reliability

The strongest reliability gain comes from the constraint layer. Problems such as malformed strings, non-JSON numbers, invalid function names, and incomplete object closures are caught before they reach output. That makes the system much more dependable in batch processing scenarios where invalid output is costly.

In short, the project trades a small amount of generation speed for a large gain in structural correctness.

## Challenges faced

A few non-trivial issues appeared during implementation:

- Tokenizer quirks: some vocabulary tokens are special tokens or multi-byte fragments and need to be filtered out.
- Function-name generation: names can share prefixes, so the decoder must allow partial matches without producing invalid names.
- JSON validity: every token must keep the partial output a valid prefix of the expected JSON object.
- Numeric edge cases: numbers such as decimals, negatives, and long values required careful validation to avoid invalid or oversized values.
- Schema matching: arguments must match the exact types declared in each function definition, not just the general structure of the JSON object.

These problems were addressed by centralizing validation in the constraint engine and restricting candidate tokens before they are accepted by the model.

## Testing strategy

The project uses a structured validation approach to make sure the implementation remains reliable:

- Prompt and schema parsing tests verify that JSON inputs are read and interpreted correctly.
- Vocabulary tests confirm that token tables are built correctly and filtered safely.
- Constraint tests validate numeric, boolean, and string behavior under partial generation.
- CLI validation checks that the entry point accepts the right arguments and produces the expected file output.

The test workflow is intentionally focused on edge cases that commonly break function-calling systems:

```bash
uv run pytest
```

This helps catch malformed outputs, missing definitions, invalid paths, and schema mismatches before they become runtime issues.

## Example usage

### Basic run

```bash
uv sync
uv run python -m src
```

### Custom input and output

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Example output shape

```json
{
  "prompt": "What is 3 sum 5?",
  "name": "add_numbers",
  "parameters": {
    "a": 3.0,
    "b": 5.0
  }
}
```

## Resources

### Related references

- OpenAI function calling documentation: https://platform.openai.com/docs/guides/function-calling
- Hugging Face guides on constrained generation and JSON decoding
- JSON Schema documentation: https://json-schema.org/
- Pydantic documentation: https://docs.pydantic.dev/
- Tokenizer and vocabulary references for local decoding pipelines

### AI usage

AI tools were used as part of the development workflow for:

- scaffolding the project structure and README layout
- explaining token-level constraints and JSON-prefix validation
- helping debug edge cases involving tokenizer quirks and numeric validation
- reviewing the logic for function-name selection and output serialization
- checking documentation clarity and project presentation for peers and recruiters

The AI assistance was used to speed up research, debugging, and documentation, while the final logic and validation decisions were still grounded in the project’s actual code and constraints.

## Summary

This project demonstrates how a small language model can be guided to produce valid function calls through constrained decoding rather than free-form generation. The goal is to combine model flexibility with strict correctness, making the system suitable for controlled tool use and structured agent-style tasks.
