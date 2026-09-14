from pydantic import BaseModel
from typing import Any, Literal


class Parameters(BaseModel):
    """Describes the expected type of a parameter."""
    type: Literal["string", "number", "boolean", "integer"]

class ReturnsParameters(BaseModel):
    type: str

class FunctionsDefinitions(BaseModel):
    """An available function, as defined in function_definitions.json."""
    name: str
    description: str
    parameters: dict[str, Parameters]
    returns: ReturnsParameters

class PromptResultats(BaseModel):
    prompt: str

class FunctionsCallResultats(BaseModel):
    """An output object in function_calling_results.json."""
    prompt: str
    name: str
    parameters: dict[str, Any]
