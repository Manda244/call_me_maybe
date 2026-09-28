"""Pydantic models for function definitions, prompts and results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TypeSpec(BaseModel):
    """Type of a parameter or of a return value."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["string", "number", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """An available function, as defined in functions_definition.json."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class PromptInput(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject prompts that contain no meaningful characters."""
        if not value.strip():
            raise ValueError("Prompt must not be empty")
        return value


# Prompt = PromptInput


class FunctionCallResult(BaseModel):
    """One output object of function_calling_results.json."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, str | float | int | bool]
