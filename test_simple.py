#!/usr/bin/env python3
"""Simple test to verify the function calling works."""
import json
from llm_sdk import Small_LLM_Model
from src.model import FunctionsDefinitions, Parameters
from src.function_selector import select_function

# Load functions
with open("data/input/functions_definition.json") as f:
    functions_data = json.load(f)

# Parse functions
functions = []
for func_data in functions_data:
    params = {
        name: Parameters(**param_def) 
        for name, param_def in func_data.get("parameters", {}).items()
    }
    func = FunctionsDefinitions(
        name=func_data["name"],
        description=func_data["description"],
        parameters=params,
        returns={"type": "string"}
    )
    functions.append(func)

print(f"Loaded {len(functions)} functions: {[f.name for f in functions]}")

# Initialize model
print("Loading model...")
model = Small_LLM_Model()
print("Model loaded!")

# Test with one prompt
test_prompt = "What is the sum of 2 and 3?"
print(f"\nTesting with prompt: {test_prompt}")

try:
    selected = select_function(model, test_prompt, functions)
    print(f"✓ Function selected: {selected.name}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTest complete!")
