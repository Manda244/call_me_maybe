from model import FunctionsDefinitions
from pydantic import TypeAdapter
import json


def load_fonction_definitions() -> list[FunctionsDefinitions]:
    """Load the function definitions from the JSON file."""
    read_file = "function_definitions.json"
    adapter = TypeAdapter(list[FunctionsDefinitions])
    try:
        with open(read_file, "r") as f:
            data = json.load(f)
            return adapter.validate_json(data)
    except FileNotFoundError:
        print(f"Error: The file '{read_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{read_file}' is not a valid JSON file.")
    finally:
        f.close()

def load_test_prompts() -> list[dict]:
    """Load the test prompts from the JSON file."""
    read_file = "test_prompts.json"
    try:
        with open(read_file, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{read_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{read_file}' is not a valid JSON file.")
    finally:
        f.close()

def write_function_calling_results(results: list[dict]):
    """Write the function calling results to a JSON file."""
    write_file = "function_calling_results.json"
    try:
        with open(write_file, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error: Could not write to the file '{write_file}'. {e}")
    finally:
        f.close()
