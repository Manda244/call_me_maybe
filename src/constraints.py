"""Constrained decoding: validate a JSON function call token by token.

The generated text must always be a valid prefix of::

    {"name":"<fn>","parameters":{"<p1>":<v1>,"<p2>":<v2>}}

where every value respects the type declared in the function definition.
"""

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models import FunctionDefinition

_HEX_DIGITS = "0123456789abcdefABCDEF"
_SIMPLE_ESCAPES = '"\\/bfnrt'
_NUMBER_CHARS = "-.0123456789"
# Exponents are not generated: they are never needed for function
# arguments and could overflow to infinity (invalid JSON once dumped).
_MAX_NUMBER_LENGTH = 30

# A prefix must always be completable into a valid JSON number,
# otherwise the decoder could end up in a dead end.
_NUMBER_PREFIX = re.compile(r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?)?")
_NUMBER_FULL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_INTEGER_PREFIX = re.compile(r"-?(?:0|[1-9][0-9]*)?")
_INTEGER_FULL = re.compile(r"-?(?:0|[1-9][0-9]*)")

_Status = Literal["invalid", "open", "closed"]


def _scan_string(body: str) -> tuple[bool, int]:
    """Scan the content of a JSON string (text after the opening quote).

    Args:
        body: Text following the opening quote.

    Returns:
        A pair ``(valid, end)``. ``valid`` is False if the text can never
        be valid JSON string content. ``end`` is the index of the closing
        quote in ``body``, or -1 if the string is not closed yet.
    """
    index = 0
    while index < len(body):
        character = body[index]
        if character == '"':
            return True, index
        if ord(character) < 32:
            return False, -1
        if character != "\\":
            index += 1
            continue
        if index + 1 >= len(body):
            return True, -1
        escaped = body[index + 1]
        if escaped == "u":
            digits = body[index + 2:index + 6]
            if any(digit not in _HEX_DIGITS for digit in digits):
                return False, -1
            if len(digits) < 4:
                return True, -1
            index += 6
        elif escaped in _SIMPLE_ESCAPES:
            index += 2
        else:
            return False, -1
    return True, -1


class DecoderState(BaseModel):
    """Text generated so far and whether the JSON object is complete."""

    model_config = ConfigDict(validate_assignment=True)

    generated: str = ""
    complete: bool = False


class JsonCallConstraint(BaseModel):
    """Constraint forcing the output to be a valid call of one function."""

    function_definition: FunctionDefinition
    function_name: str
    state: DecoderState = Field(default_factory=DecoderState)

    def _opening(self) -> str:
        """Return the fixed JSON prefix for the selected function."""
        name = json.dumps(self.function_name, ensure_ascii=False)
        return '{"name":' + name + ',"parameters":{'

    def _key_prefix(self, parameter_name: str, index: int) -> str:
        """Return the fixed text preceding the value of one parameter."""
        separator = "," if index > 0 else ""
        key = json.dumps(parameter_name, ensure_ascii=False)
        return separator + key + ":"

    def _consume_value(
        self, text: str, value_type: str
    ) -> tuple[_Status, str]:
        """Consume one value from the start of ``text``.

        Returns:
            ``("invalid", "")`` if the text cannot be a valid value,
            ``("open", "")`` if the text ends inside the value, or
            ``("closed", rest)`` if the value is finished, ``rest`` being
            the text that follows it.
        """
        if value_type == "string":
            return self._consume_string(text)
        if value_type in ("number", "integer"):
            return self._consume_number(text, value_type == "integer")
        return self._consume_boolean(text)

    def _consume_string(self, text: str) -> tuple[_Status, str]:
        """Consume a JSON string value."""
        if not text:
            return "open", ""
        if text[0] != '"':
            return "invalid", ""
        valid, end = _scan_string(text[1:])
        if not valid:
            return "invalid", ""
        if end < 0:
            return "open", ""
        return "closed", text[end + 2:]

    def _consume_number(
        self, text: str, integer: bool
    ) -> tuple[_Status, str]:
        """Consume a JSON number (or integer) value."""
        prefix = _INTEGER_PREFIX if integer else _NUMBER_PREFIX
        full = _INTEGER_FULL if integer else _NUMBER_FULL
        end = 0
        while end < len(text) and text[end] in _NUMBER_CHARS:
            end += 1
        value = text[:end]
        if len(value) > _MAX_NUMBER_LENGTH:
            return "invalid", ""
        if end == len(text):
            # An incomplete number may not reach the maximum length,
            # otherwise it could never be completed.
            room = len(value) < _MAX_NUMBER_LENGTH
            if prefix.fullmatch(value) and (room or full.fullmatch(value)):
                return "open", ""
            return "invalid", ""
        if full.fullmatch(value):
            return "closed", text[end:]
        return "invalid", ""

    def _consume_boolean(self, text: str) -> tuple[_Status, str]:
        """Consume a JSON boolean value."""
        for literal in ("true", "false"):
            if text.startswith(literal):
                return "closed", text[len(literal):]
            if literal.startswith(text):
                return "open", ""
        return "invalid", ""

    def is_valid_prefix(self, text: str) -> bool:
        """Return whether ``text`` is a valid prefix of the constrained call.

        Args:
            text: Full generated text, including the candidate token.
        """
        opening = self._opening()
        if len(text) <= len(opening):
            return opening.startswith(text)
        if not text.startswith(opening):
            return False
        rest = text[len(opening):]
        parameters = self.function_definition.parameters
        for index, (name, spec) in enumerate(parameters.items()):
            prefix = self._key_prefix(name, index)
            if len(rest) <= len(prefix):
                return prefix.startswith(rest)
            if not rest.startswith(prefix):
                return False
            status, rest = self._consume_value(
                rest[len(prefix):], spec.type
            )
            if status == "invalid":
                return False
            if status == "open":
                return True
        return "}}".startswith(rest)

    def is_valid_token(self, token_text: str) -> bool:
        """Return whether appending ``token_text`` keeps the JSON valid."""
        if not token_text:
            return False
        return self.is_valid_prefix(self.state.generated + token_text)

    def select_token(
        self, ranked_ids: Iterable[int], token_texts: Mapping[int, str]
    ) -> int:
        """Return the best-ranked token id that respects the constraint.

        This is equivalent to setting invalid logits to ``-inf`` and taking
        the argmax, but only tests candidates until one is valid.

        Args:
            ranked_ids: Token ids sorted by decreasing logit.
            token_texts: Mapping from token id to its literal text.

        Raises:
            ValueError: If no token satisfies the constraint.
        """
        for token_id in ranked_ids:
            if self.is_valid_token(token_texts.get(token_id, "")):
                return int(token_id)
        raise ValueError("No vocabulary token satisfies the JSON schema")

    def get_allowed_tokens(
        self, vocab: Mapping[str, int]
    ) -> list[tuple[int, str]]:
        """Return all (id, text) pairs allowed at this step (slow).

        Scans the whole vocabulary; prefer ``select_token`` in the
        generation loop.

        Raises:
            ValueError: If no token satisfies the constraint.
        """
        allowed = [
            (token_id, token_text)
            for token_text, token_id in vocab.items()
            if self.is_valid_token(token_text)
        ]
        if not allowed:
            raise ValueError("No vocabulary token satisfies the JSON schema")
        return allowed

    def consume(self, token_text: str) -> None:
        """Append a token after checking it preserves the schema.

        Raises:
            ValueError: If the token would break the schema.
        """
        candidate = self.state.generated + token_text
        if not self.is_valid_prefix(candidate):
            raise ValueError("Token violates the JSON schema")
        self.state.generated = candidate
        if candidate.endswith("}}") and self._is_complete(candidate):
            self.state.complete = True

    def _is_complete(self, text: str) -> bool:
        """Return whether ``text`` is a complete call object."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        if set(payload) != {"name", "parameters"}:
            return False
        parameters = payload["parameters"]
        expected = set(self.function_definition.parameters)
        return isinstance(parameters, dict) and set(parameters) == expected

    def is_complete(self) -> bool:
        """Return whether the constrained object is complete."""
        return self.state.complete

    def resolved_parameters(self) -> dict[str, str | float | int | bool]:
        """Return the generated parameters with the declared types.

        Numbers are converted to ``float`` as in the expected output
        (``{"a": 2.0}``).

        Raises:
            ValueError: If the generation is incomplete or a value does
                not match its declared type.
        """
        if not self.is_complete():
            raise ValueError("JSON generation is incomplete")
        payload = json.loads(self.state.generated)
        raw = payload.get("parameters")
        if not isinstance(raw, dict):
            raise ValueError("Generated parameters are not an object")
        result: dict[str, str | float | int | bool] = {}
        for name, spec in self.function_definition.parameters.items():
            if name not in raw:
                raise ValueError(f"Missing parameter: {name}")
            result[name] = self._convert(name, raw[name], spec.type)
        return result

    @staticmethod
    def _convert(
        name: str, value: object, value_type: str
    ) -> str | float | int | bool:
        """Check a decoded value against its type and normalize it."""
        if value_type == "string" and isinstance(value, str):
            return value
        if value_type == "boolean" and isinstance(value, bool):
            return value
        is_numeric = isinstance(value, (int, float))
        if is_numeric and not isinstance(value, bool):
            if value_type == "integer" and isinstance(value, int):
                return value
            if value_type == "number":
                try:
                    number = float(value)  # type: ignore[arg-type]
                except OverflowError:
                    raise ValueError(f"Number too large: {name}") from None
                if not math.isfinite(number):
                    raise ValueError(f"Number too large: {name}")
                return number
        raise ValueError(f"Parameter {name!r} is not a valid {value_type}")
