"""Vocabulary loading and token/bytes conversion for constrained decoding."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from src.io_utils import read_json_file


def _bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 byte -> printable unicode character mapping.

    Byte-level BPE gives each of the 256 possible bytes a visible
    character. Printable bytes map to themselves; the others (space,
    newline, control bytes...) are shifted to 256 and above, which is
    why a space appears as 'G-dot' (U+0120) in the vocabulary.
    """
    kept = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = kept[:]
    extra = 0
    for byte in range(256):
        if byte not in kept:
            kept.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(kept, codes)}


_CHAR_TO_BYTE: dict[str, int] = {
    char: byte for byte, char in _bytes_to_unicode().items()
}


def token_to_bytes(token: str) -> bytes:
    """Convert a raw BPE vocabulary token into the bytes it stands for.

    Characters missing from the byte table (special tokens such as
    ``<|endoftext|>`` may contain some) fall back to their UTF-8
    encoding.
    """
    output = bytearray()
    for char in token:
        if char in _CHAR_TO_BYTE:
            output.append(_CHAR_TO_BYTE[char])
        else:
            output.extend(char.encode("utf-8"))
    return bytes(output)


class Vocabulary(BaseModel):
    """Tokenizer vocabulary: token id <-> raw token <-> bytes.

    Build it with ``Vocabulary.from_file(path)`` where ``path`` comes
    from ``Small_LLM_Model.get_path_to_vocab_file()``.

    Attributes:
        vocab_path: Path of the file the vocabulary was loaded from.
        token_to_id: Raw BPE token string -> token id.
        id_to_token: Token id -> raw BPE token string.
        token_bytes: Token id -> bytes the token writes. Precomputed
            once so the masking loop never converts strings.
    """

    model_config = ConfigDict(frozen=True)

    vocab_path: Path
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    token_bytes: dict[int, bytes]

    @classmethod
    def from_file(cls, vocab_path: str | Path) -> "Vocabulary":
        """Load, validate and index a vocabulary JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read.
            ValueError: If the file is not valid JSON or is not a
                flat object of {token: integer id}.
        """
        path = Path(vocab_path)
        raw = read_json_file(path)
        try:
            token_to_id = TypeAdapter(dict[str, int]).validate_python(
                raw, strict=True
            )
        except ValidationError as exc:
            raise ValueError(
                f"Vocabulary must map token strings to integer ids: {path}"
            ) from exc
        id_to_token = {i: tok for tok, i in token_to_id.items()}
        token_bytes = {i: token_to_bytes(t) for i, t in id_to_token.items()}
        return cls(
            vocab_path=path,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            token_bytes=token_bytes,
        )

    @property
    def size(self) -> int:
        """Return the highest known token id + 1."""
        return max(self.id_to_token) + 1 if self.id_to_token else 0

    def get_token(self, token_id: int) -> str:
        """Return the raw BPE token string for a token id.

        Raises:
            KeyError: If the id is not in the vocabulary.
        """
        if token_id not in self.id_to_token:
            raise KeyError(f"Token id not found in vocabulary: {token_id}")
        return self.id_to_token[token_id]

    def get_id(self, token: str) -> int:
        """Return the token id for a raw BPE token string.

        Raises:
            KeyError: If the token is not in the vocabulary.
        """
        if token not in self.token_to_id:
            raise KeyError(f"Token not found in vocabulary: {token}")
        return self.token_to_id[token]

    def get_bytes(self, token_id: int) -> bytes:
        """Return the bytes written by a token id.

        Raises:
            KeyError: If the id is not in the vocabulary.
        """
        if token_id not in self.token_bytes:
            raise KeyError(f"Token id not found in vocabulary: {token_id}")
        return self.token_bytes[token_id]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids to text.

        Bytes of all tokens are joined first and decoded as UTF-8 once,
        because a multi-byte character can be split across tokens.
        Invalid sequences become U+FFFD instead of raising.
        """
        data = b"".join(self.get_bytes(i) for i in token_ids)
        return data.decode("utf-8", errors="replace")
