"""Vocabulary loading and token-id/text conversion for constrained decoding."""

import json
from pathlib import Path


class Vocabulary:
    """Loads a tokenizer vocabulary file and maps between token ids and text.

    The vocabulary file (as returned by
    ``Small_LLM_Model.get_path_to_vocab_file()``) is a JSON object
    mapping raw BPE token strings to their integer ids. This class
    keeps both directions (token -> id and id -> token) and knows how
    to turn a raw BPE token into the literal text it represents.
    """

    def __init__(self, vocab_path: str | Path) -> None:
        """Load and index the vocabulary.

        Args:
            vocab_path: Path to the vocabulary JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON or does not
                contain a flat object of {token: id}.
        """
        self.vocab_path = Path(vocab_path)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        """Read the vocabulary file and populate the lookup tables."""
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
        try:
            raw = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid vocabulary JSON: {self.vocab_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Vocabulary must be a JSON object: {self.vocab_path}")
        for token, token_id in raw.items():
            if not isinstance(token, str):
                raise ValueError("Vocabulary keys must be strings")
            if not isinstance(token_id, int):
                raise ValueError("Vocabulary values must be integer token ids")
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def get_token(self, token_id: int) -> str:
        """Return the raw BPE token string for a given token id.

        Raises:
            KeyError: If the id is not present in the vocabulary.
        """
        if token_id not in self.id_to_token:
            raise KeyError(f"Token id not found in vocabulary: {token_id}")
        return self.id_to_token[token_id]

    def get_id(self, token: str) -> int:
        """Return the token id for a given raw BPE token string.

        Raises:
            KeyError: If the token is not present in the vocabulary.
        """
        if token not in self.token_to_id:
            raise KeyError(f"Token string not found in vocabulary: {token}")
        return self.token_to_id[token]

    def token_to_text(self, token: str) -> str:
        """Convert a raw BPE vocabulary token into its literal text.

        Byte-level BPE tokenizers (as used by Qwen) encode a leading
        space as 'Ġ' (U+0120) and a newline as 'Ċ' (U+010A). This
        replaces those markers with the actual characters they stand
        for, so the result matches what `decode()` would produce.

        Note: this only handles the two markers relevant to the
        characters used in this project (letters, digits, JSON
        punctuation). It is not a full byte-to-unicode reversal of
        GPT-2-style BPE, which also remaps a handful of rare control
        bytes not needed here.
        """
        return token.replace("\u0120", " ").replace("\u010a", "\n")
