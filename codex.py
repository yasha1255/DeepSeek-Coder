from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda"
device="xpu"
if device == "xpu":
    print(torch.xpu.is_available())

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)

# Load the model, specifying the device_map and torch_dtype for XPU
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device)

def codex(code="", max_length=128):
    inputs = tokenizer(code, return_tensors="pt").to(device)

    # Generate the output
    outputs = model.generate(**inputs, max_length=max_length)

    # Decode and print the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(code):]