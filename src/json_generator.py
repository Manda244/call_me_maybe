import json
from llm_sdk import Small_LLM_Model

from src.function_selector import _pick_best_allowed_token
from .model import FunctionsDefinitions


def generate_parameters(model: Small_LLM_Model, prompt: str, function: FunctionsDefinitions) -> dict:
    # Convert Pydantic models to dicts for JSON serialization
    params_dict = {k: v.model_dump() for k, v in function.parameters.items()}
    context = (
        f"Extract arguments for the function '{function.name}' from the following prompt: {prompt}"
        f"Parameters: {list(function.parameters.keys())}"
        f"json format: {json.dumps(params_dict)}"
    )
    input_ids = model.encode(context)
    vocab = _load_vocab(model)

    state = _JsonState(params_dict)
    generated: list[int] = []

    max_generation_steps = 200  # Prevent infinite loops
    while not state.is_complete() and len(generated) < max_generation_steps:
        allowed_ids = state.allowed_token_ids(vocab)
        # Convert tensor to list if needed
        if hasattr(input_ids, 'tolist'):
            input_ids_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        else:
            input_ids_list = input_ids
        logits = model.get_logits_from_input_ids(input_ids_list + generated)
        next_token_id = _pick_best_allowed_token(logits, allowed_ids)
        generated.append(next_token_id)
        state.consume(vocab[next_token_id])

    text = "".join(state.emitted_tokens)
    return json.loads(text) if text else {}

def _load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load the model's vocabulary as a mapping from token ids to strings."""
    path = model.get_path_to_vocab_file()
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    # Create a reverse mapping from token ids to strings
    return {v: k for k, v in vocab.items()}

class _JsonState:
    """A simple state machine to track the generation of a JSON object."""

    def __init__(self, schema: dict):
        self.schema = schema
        self.emitted_tokens: list[str] = []
        self.current_key: str | None = None
        self.current_value: str | None = None
        self.stack: list[dict] = [schema]

    def is_complete(self) -> bool:
        return len(self.stack) == 0

    def allowed_token_ids(self, vocab: dict[int, str]) -> set[int]:
        """Return the set of token ids that are valid next tokens."""
        # Return all valid token IDs from the vocabulary
        # This is a simplified approach - in production you'd want smarter constraints
        return set(vocab.keys())
    
    def _token_ids_for_string(self, s: str, vocab: dict[int, str]) -> set[int]:
        """Get all token IDs that could represent this string or parts of it."""
        token_ids = set()
        # Look for exact matches and partial matches
        for token_id, token_str in vocab.items():
            # Exact match
            if token_str == s:
                token_ids.add(token_id)
            # Token starts with the string
            elif token_str.startswith(s):
                token_ids.add(token_id)
            # The string contains the token (useful for building)
            elif s.startswith(token_str) and len(token_str) > 1:
                token_ids.add(token_id)
        
        # If no matches found, return any reasonable token (fallback)
        if not token_ids:
            # Return a broader set of tokens that might work
            for token_id, token_str in vocab.items():
                if len(token_str) <= 1 or token_str.isalnum():
                    token_ids.add(token_id)
                    if len(token_ids) >= 10:  # Limit to avoid too many options
                        break
        
        return token_ids if token_ids else set(vocab.keys())
    
    def _token_id_for_string(self, s: str, vocab: dict[int, str]) -> int:
        """Get a token ID for a string (find the first matching token)."""
        # First try exact match
        for token_id, token_str in vocab.items():
            if token_str == s:
                return token_id
        
        # Then try tokens that start with the string
        for token_id, token_str in vocab.items():
            if token_str.startswith(s):
                return token_id
        
        # Then try any token that's part of the string
        for token_id, token_str in vocab.items():
            if token_str in s:
                return token_id
        
        # Fallback: return any valid token ID
        if vocab:
            return next(iter(vocab.keys()))
        raise ValueError(f"No token found for string: {s}")
    
    def consume(self, token_str: str) -> None:
        """Process a token and update the state machine."""
        self.emitted_tokens.append(token_str)
        # Simple state machine logic: track whether we're building a key or value
        if self.current_key is None and token_str != "{":
            # Try to identify if this is a key
            if token_str not in [" ", ",", "{", "}", ":"]:
                self.current_key = token_str
        elif self.current_key is not None and token_str == ":":
            # We've found a separator, now expecting a value
            self.current_value = ""
        elif self.current_value is not None:
            if token_str == ",":
                # End of this key-value pair
                self.current_key = None
                self.current_value = None
            elif token_str == "}":
                # End of this object
                self.stack.pop()
                self.current_key = None
                self.current_value = None
            else:
                # Accumulate the value
                self.current_value += token_str
