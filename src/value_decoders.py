import json
import re

from pydantic import BaseModel, ConfigDict, Field


class NumberValueDecoder(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    stop_tokens: set[str] = Field(default_factory=set)
    allow_decimal: bool = True
    value: str = ""
    done: bool = False

    def _is_valid_extension(self, token_text: str) -> bool:
        """Return whether a token keeps a valid JSON number prefix."""
        candidate = self.value + token_text
        if not candidate or any(
            character not in "-+.eE0123456789" for character in candidate
        ):
            return False
        if not self.allow_decimal and any(
            character in candidate for character in ".eE"
        ):
            return False
        if candidate in {"-", "+", ".", "-."}:
            return candidate == "-"
        pattern = (
            r"-?(?:0|[1-9][0-9]*)?(?:\.[0-9]*)?"
            r"(?:[eE][+-]?[0-9]*)?"
        )
        if re.fullmatch(pattern, candidate) is None:
            return False
        return any(character.isdigit() for character in candidate)

    def _is_complete_value(self) -> bool:
        """Return whether the current text is a complete JSON number."""
        pattern = (
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
            r"(?:[eE][+-]?[0-9]+)?"
        )
        if re.fullmatch(pattern, self.value) is None:
            return False
        if not self.allow_decimal and any(
            character in self.value for character in ".eE"
        ):
            return False
        return True

    def get_allowed_tokens(
        self, vocab: dict[str, int]
    ) -> list[tuple[int, str]]:
        """Return vocabulary tokens that preserve a valid numeric state."""
        allowed: list[tuple[int, str]] = []
        for token_text, token_id in vocab.items():
            if token_text in self.stop_tokens and self._is_complete_value():
                allowed.append((token_id, token_text))
            elif self._is_valid_extension(token_text):
                allowed.append((token_id, token_text))
        if not allowed:
            raise ValueError("No token can extend the number")
        return allowed

    def consume(self, token_text: str) -> None:
        """Consume a numeric token or a terminating separator."""
        if token_text in self.stop_tokens:
            if not self._is_complete_value():
                raise ValueError("Cannot terminate an incomplete number")
            self.done = True
            return
        if not self._is_valid_extension(token_text):
            raise ValueError("Token is not a valid numeric extension")
        self.value += token_text

    def is_complete(self) -> bool:
        """Return whether a separator has terminated the number."""
        return self.done

    def resolved_value(self) -> int | float:
        """Return the generated number using its narrowest JSON type."""
        if not self.done or not self._is_complete_value():
            raise ValueError("Number generation is incomplete")
        if any(character in self.value for character in ".eE"):
            return float(self.value)
        return int(self.value)


class StringValueDecoder(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    stop_tokens: set[str] = Field(default_factory=set)
    value: str = ""
    done: bool = False

    def _is_valid_fragment(self, token_text: str) -> bool:
        """Return whether a token is valid JSON string content."""
        candidate = self.value + token_text
        escaped = False
        for character in candidate:
            if escaped:
                if character not in '"\\/bfnrtu0123456789abcdefABCDEF':
                    return False
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"' or ord(character) < 32:
                return False
        return True

    def _is_complete_fragment(self) -> bool:
        """Return whether the string content is not ending in an escape."""
        return not self.value.endswith("\\")

    def get_allowed_tokens(
        self, vocab: dict[str, int]
    ) -> list[tuple[int, str]]:
        """Return vocabulary tokens that preserve a valid JSON string state."""
        allowed: list[tuple[int, str]] = []
        for token_text, token_id in vocab.items():
            if token_text in self.stop_tokens and self._is_complete_fragment():
                allowed.append((token_id, token_text))
            elif token_text and self._is_valid_fragment(token_text):
                allowed.append((token_id, token_text))
        if not allowed:
            raise ValueError("No token can extend the string")
        return allowed

    def consume(self, token_text: str) -> None:
        """Consume string content or mark the value complete on a quote."""
        if token_text in self.stop_tokens:
            if not self._is_complete_fragment():
                raise ValueError("Cannot terminate an escaped string")
            self.done = True
            return
        if not self._is_valid_fragment(token_text):
            raise ValueError("Token is not valid JSON string content")
        self.value += token_text

    def is_complete(self) -> bool:
        """Return whether a closing quote has terminated the string."""
        return self.done

    def resolved_value(self) -> str:
        """Decode the generated JSON string content."""
        if not self.done:
            raise ValueError("String generation is incomplete")
        try:
            return str(json.loads('"' + self.value + '"'))
        except json.JSONDecodeError as exc:
            raise ValueError("Generated string is not valid JSON") from exc


class BooleanValueDecoder(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    stop_tokens: set[str] = Field(default_factory=set)
    value: str = ""
    done: bool = False

    def get_allowed_tokens(
        self, vocab: dict[str, int]
    ) -> list[tuple[int, str]]:
        """Return tokens that preserve either JSON boolean literal."""
        allowed: list[tuple[int, str]] = []
        for token_text, token_id in vocab.items():
            if token_text in self.stop_tokens and self.value in {
                "true", "false"
            }:
                allowed.append((token_id, token_text))
                continue
            if token_text and any(
                literal.startswith(self.value + token_text)
                for literal in ("true", "false")
            ):
                allowed.append((token_id, token_text))
        if not allowed:
            raise ValueError("No token can extend the boolean")
        return allowed

    def consume(self, token_text: str) -> None:
        """Consume a boolean literal or mark it complete on a separator."""
        if token_text in self.stop_tokens:
            if self.value not in {"true", "false"}:
                raise ValueError("Cannot terminate an incomplete boolean")
            self.done = True
            return
        allowed = [
            literal
            for literal in ("true", "false")
            if literal.startswith(self.value + token_text)
        ]
        if not allowed:
            raise ValueError("Token is not a valid boolean extension")
        self.value += token_text

    def is_complete(self) -> bool:
        """Return whether a separator has terminated the boolean."""
        return self.done

    def resolved_value(self) -> bool:
        """Return the generated boolean value."""
        if not self.done or self.value not in {"true", "false"}:
            raise ValueError("Boolean generation is incomplete")
        return self.value == "true"
