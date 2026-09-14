import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .model import FunctionsDefinitions, PromptResultats, FunctionsCallResultats
from .function_selector import select_function
from .json_generator import generate_parameters
from llm_sdk import Small_LLM_Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    return parser.parse_args()

def load_json_file(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    args = parse_args()

    raw_functions = load_json_file(args.functions_definition)
    raw_prompts = load_json_file(args.input)

    try:
        functions = [FunctionsDefinitions(**f) for f in raw_functions]
        prompts = [PromptResultats(**p) for p in raw_prompts]
    except ValidationError as e:
        print(f"Error: schema validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    model = Small_LLM_Model()

    results: list[FunctionsCallResultats] = []
    for entry in prompts:
        try:
            function = select_function(model, entry.prompt, functions)
            parameters = generate_parameters(model, entry.prompt, function)
            results.append(FunctionsCallResultats(prompt=entry.prompt, name=function.name, parameters=parameters))
        except Exception as e:
            print(f"Error processing prompt '{entry.prompt}': {e}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=4)


if __name__ == "__main__":
    main()
