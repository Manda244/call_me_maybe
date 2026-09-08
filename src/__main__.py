from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

marasolo = "What is the sum of 2 and 3?"
ids = model.encode(marasolo).tolist()[0]
print(ids)
print(marasolo)

nb_tokens_after = len(ids)
print(f"Number of tokens after encoding: {nb_tokens_after}")

generated_ids = []

for _ in range(nb_tokens_after):
    test = model.get_logits_from_input_ids(ids)
    test_next = test.index(max(test))
    generated_ids.append(test_next)
    ids.append(test_next)
    print(f"Next token id: {test_next}, Token: {model.decode([test_next])}")

print(generated_ids)
print(model.decode(generated_ids))
