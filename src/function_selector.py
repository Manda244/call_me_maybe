import torch
from llm_sdk import Small_LLM_Model
from .model import FunctionsDefinitions


def _pick_best_allowed_token(logits: list[float], allowed_tokens: set[int]) -> int:
    """Pick the token with the highest logit from the allowed tokens."""
    # Filter to only tokens that are in allowed_tokens and within logits range
    valid_tokens = [token for token in allowed_tokens if 0 <= token < len(logits)]
    
    if not valid_tokens:
        # Fallback: pick the token with highest logit overall
        # This can happen if all allowed_tokens are out of range
        return max(range(len(logits)), key=lambda i: logits[i])
    
    # Pick the best among valid tokens
    filtered_logits = {token: logits[token] for token in valid_tokens}
    return max(filtered_logits, key=filtered_logits.get)

def _to_tensor(token_ids: list[int]) -> list[int]:
    """Convert a list of token ids to a tensor-like structure (placeholder)."""
    return torch.tensor([token_ids])  # In a real implementation, this would convert to a tensor type

def _constrained_choice(model: Small_LLM_Model, input_ids: list[int], valid_names: list[str]) -> str:
    """Generate token-by-token, restricting output to one of choices."""
    # Encode each choice into token id sequences
    choice_token_ids = {c: model.encode(c) for c in valid_names}
    # Convert all tensors to lists for easier comparison
    choice_token_lists = {}
    for choice, ids in choice_token_ids.items():
        if hasattr(ids, 'squeeze'):
            ids_list = ids.squeeze().tolist() if ids.dim() > 1 else ids.tolist()
        else:
            ids_list = ids
        choice_token_lists[choice] = ids_list
    
    generated: list[int] = []
    remaining = set(choice_token_lists.keys())

    max_len = max(len(ids) for ids in choice_token_lists.values())
    max_steps = max_len + 10  # Add some buffer for safety

    for step in range(max_steps):
        # Get the next token probabilities
        logits = model.get_logits_from_input_ids(input_ids + generated)
        # Filter logits to only allow valid next tokens
        allowed_tokens = set()
        for choice in remaining:
            ids_list = choice_token_lists[choice]
            if len(ids_list) > step:
                allowed_tokens.add(ids_list[step])
        
        # If no more valid choices, return any remaining one
        if not allowed_tokens:
            if remaining:
                return next(iter(remaining))
            raise ValueError("No valid function name could be generated.")
        
        # Filter to valid token IDs (within vocab range)
        valid_tokens = [token for token in allowed_tokens if 0 <= token < len(logits)]
        if not valid_tokens:
            # Fallback: if no valid tokens from constraints, pick highest logit overall
            if remaining:
                return next(iter(remaining))
            next_token = max(range(len(logits)), key=lambda i: logits[i])
        else:
            filtered_logits = {token: logits[token] for token in valid_tokens}
            next_token = max(filtered_logits, key=filtered_logits.get)
        generated.append(next_token)

        # Check if any choice has been fully generated or is no longer valid
        still_remaining = set()
        for choice in remaining:
            ids_list = choice_token_lists[choice]
            if step + 1 >= len(ids_list):
                # This choice is complete
                if generated == ids_list:
                    return choice
            elif generated == ids_list[:step + 1]:
                # This choice is still possible
                still_remaining.add(choice)
        
        remaining = still_remaining
        if not remaining:
            raise ValueError("No valid function name could be generated.")

    if remaining:
        # Return the only remaining choice
        return next(iter(remaining))
    raise ValueError("Could not generate a valid function name from the model output.")

def select_function(
        model: Small_LLM_Model,
        prompt: str,
        functions: list[FunctionsDefinitions]
) -> FunctionsDefinitions:
    """Ask the LLM to choose a function name, limiting it to valid names only."""
    valid_names = []
    for function in functions:
        valid_names.append(function.name)

    # build a prompt description available functions
    context =  "\n".join(f"- {function.name}: {function.description}" for function in functions)
    full_prompt = (
        f"Functions: {context}"
        f"Users request: {prompt}"
        f"Which function should be called? Please respond with the function name only, and make sure it is one of the valid names: {valid_names}"
    )

    input_ids_tensor = model.encode(full_prompt)
    # Convert tensor to list
    input_ids_list = input_ids_tensor[0].tolist() if input_ids_tensor.dim() > 1 else input_ids_tensor.tolist()
    chosen = _constrained_choice(model, input_ids_list, valid_names)

    for f in functions:
        if f.name == chosen:
            return f
    raise ValueError("Chosen function not found")
