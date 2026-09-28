import json
import re

from pydantic import BaseModel, ConfigDict

from src.models import FunctionDefinition


class DecoderState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    generated: str = ""
    complete: bool = False


class JsonCallConstraint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    function_definition: FunctionDefinition
    function_name: str
    state: DecoderState = DecoderState()

    def _opening(self) -> str:
        """Return the fixed JSON prefix for the selected function."""
        return (
            '{"name":"'
            + self.function_name
            + '\",\"parameters\":{'
        )

    def _value_start(self, parameter_name: str, index: int) -> str:
        """Return the fixed key prefix for one parameter."""
        separator = ""
        if index != 0:
            separator = ","
        return separator + '"' + parameter_name + '":'

    def _valid_string_prefix(self, value: str) -> bool:
        """Return whether a string fragment is valid JSON content."""
        escaped = False
        for character in value:
            if escaped:
                if character not in '"\\/bfnrtu0123456789abcdefABCDEF':
                    return False
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"' or ord(character) < 32:
                return False
        return True

    def _valid_number_prefix(self, value: str) -> bool:
        """Return whether a fragment is a valid JSON number prefix."""
        if not value:
            return True
        if any(character not in "-+.eE0123456789" for character in value):
            return False
        pattern = (
            r"-?(?:0|[1-9][0-9]*)?(?:\.[0-9]*)?"
            r"(?:[eE][+-]?[0-9]*)?"
        )
        if re.fullmatch(pattern, value) is None:
            return False
        return value == "-" or any(character.isdigit() for character in value)

    def _valid_complete_number(self, value: str) -> bool:
        """Return whether a fragment is a complete JSON number."""
        pattern = (
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
            r"(?:[eE][+-]?[0-9]+)?"
        )
        return re.fullmatch(pattern, value) is not None

    def _valid_boolean_prefix(self, value: str) -> bool:
        """Return whether a fragment prefixes a JSON boolean literal."""
        return value == "" or any(
            literal.startswith(value) for literal in ("true", "false")
        )

    def _parse_value_prefix(
        self, text: str, parameter_index: int, offset: int
    ) -> bool:
        """Validate one value and recursively validate its following fields."""
        parameter_items = list(self.function_definition.parameters.items())
        parameter_type = parameter_items[parameter_index][1].type
        remainder = text[offset:]
        if parameter_type == "string":
            return self._parse_string(remainder, parameter_index)
        if parameter_type in {"number", "integer"}:
            return self._parse_number(remainder, parameter_index, parameter_type)
        return self._parse_boolean(remainder, parameter_index)

    def _parse_string(self, remainder: str, parameter_index: int) -> bool:
        """Validate a string value and its structural suffix."""
        if not remainder:
            return True
        if not remainder.startswith('"'):
            return remainder == '"'[:len(remainder)]
        remainder = remainder[1:]
        escaped = False
        for index, character in enumerate(remainder):
            if escaped:
                if character not in '"\\/bfnrtu0123456789abcdefABCDEF':
                    return False
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                return self._parse_suffix(remainder[index + 1:], parameter_index)
            if ord(character) < 32:
                return False
        return self._valid_string_prefix(remainder)

    def _parse_number(
        self, remainder: str, parameter_index: int, parameter_type: str
    ) -> bool:
        """Validate a numeric value and its structural suffix."""
        for index, character in enumerate(remainder):
            if character in "-+.eE0123456789":
                continue
            value = remainder[:index]
            if not self._valid_complete_number(value):
                return False
            if parameter_type == "integer" and any(
                marker in value for marker in ".eE"
            ):
                return False
            return self._parse_suffix(remainder[index:], parameter_index)
        if not self._valid_number_prefix(remainder):
            return False
        if parameter_type == "integer" and any(
            marker in remainder for marker in ".eE"
        ):
            return False
        return True

    def _parse_boolean(self, remainder: str, parameter_index: int) -> bool:
        """Validate a boolean value and its structural suffix."""
        for literal in ("true", "false"):
            if literal.startswith(remainder):
                return True
            if remainder.startswith(literal):
                return self._parse_suffix(
                    remainder[len(literal):], parameter_index
                )
        return False

    def _parse_suffix(self, remainder: str, parameter_index: int) -> bool:
        """Validate the next parameter prefix or the closing brace."""
        parameter_items = list(self.function_definition.parameters.items())
        if parameter_index + 1 == len(parameter_items):
            if not remainder:
                return True
            return "}}".startswith(remainder)
        next_name = parameter_items[parameter_index + 1][0]
        next_prefix = self._value_start(next_name, parameter_index + 1)
        if next_prefix.startswith(remainder):
            return True
        if remainder.startswith(next_prefix):
            return self._parse_value_prefix(
                remainder, parameter_index + 1, len(next_prefix)
            )
        return False

    def is_valid_prefix(self, text: str) -> bool:
        """Return whether text is a valid prefix of the constrained JSON call."""
        opening = self._opening()
        if len(text) <= len(opening) and opening.startswith(text):
            return True
        if not text.startswith(opening):
            return False
        parameter_items = list(self.function_definition.parameters.items())
        if not parameter_items:
            return "}}".startswith(text[len(opening):])
        offset = len(opening)
        for index, (parameter_name, _) in enumerate(parameter_items):
            prefix = self._value_start(parameter_name, index)
            remaining = text[offset:]
            if len(remaining) < len(prefix) and prefix.startswith(remaining):
                return True
            if not remaining.startswith(prefix):
                return False
            offset += len(prefix)
            return self._parse_value_prefix(text, index, offset)
        return False

    def get_allowed_tokens(self, vocab: dict[str, int]) -> list[tuple[int, str]]:
        """Return token ids whose literal text preserves the JSON constraint."""
        allowed: list[tuple[int, str]] = []
        for token_text, token_id in vocab.items():
            if token_text and self.is_valid_prefix(self.state.generated + token_text):
                allowed.append((token_id, token_text))
        if not allowed:
            raise ValueError("No vocabulary token satisfies the JSON schema")
        return allowed

    def consume(self, token_text: str) -> None:
        """Append a token after checking it preserves the schema."""
        candidate = self.state.generated + token_text
        if not self.is_valid_prefix(candidate):
            raise ValueError("Token violates the JSON schema")
        self.state.generated = candidate
        if candidate.endswith("}") and self._is_complete(candidate):
            self.state.complete = True

    def _is_complete(self, text: str) -> bool:
        """Return whether the generated text is a complete JSON object."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and set(payload) == {"name", "parameters"}

    def is_complete(self) -> bool:
        """Return whether the constrained object is complete."""
        return self.state.complete

    def resolved_parameters(self) -> dict[str, object]:
        """Return the generated parameter object after final JSON validation."""
        if not self.is_complete():
            raise ValueError("JSON generation is incomplete")
        payload = json.loads(self.state.generated)
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("Generated parameters are not an object")
        return parameters
