import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
chat_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(chat_input, max_new_tokens=50)
# Decode and print the entire assistant output, including special tokens and formatting
assistant_output = tokenizer.decode(chat_outputs[0][chat_input.shape[-1]:], skip_special_tokens=False)
print(assistant_output)
