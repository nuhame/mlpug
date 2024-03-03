from transformers import GPT2Tokenizer, TFGPT2Model

SPECIAL_TOKENS_MAPPING = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'additional_special_tokens': ['<speaker1>', '<speaker2>']
}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2Model.from_pretrained("gpt2")

print("Evaluating TFGPT2Model BEFORE extending the tokenizer and model with additional tokens ...")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
print(f"inputs = \n{inputs}\n")

outputs = model(inputs)
print(f"DONE!")

print("Adding tokens...")
orig_num_tokens = len(tokenizer)
num_special_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAPPING)
print(f"orig_num_tokens = {orig_num_tokens}, num_special_tokens={num_special_tokens}")

model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_special_tokens)

print("Evaluating TFGPT2Model AFTER extending the tokenizer and model with additional tokens ...")

inputs = tokenizer("<speaker1>Hello, my dog is cute<speaker2>I agree!", return_tensors="tf")
print(f"inputs = \n{inputs}\n")

outputs = model(inputs)
print(f"DONE!")

# print("Evaluating TFGPT2DoubleHeadsModel ...")
# model = TFGPT2DoubleHeadsModel.from_pretrained("gpt2")
# model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_special_tokens)
#
# outputs = model(inputs)



