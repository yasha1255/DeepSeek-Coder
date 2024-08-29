from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import time

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
    start_at = time.time()
    # Tokenize the
    inputs = tokenizer(code, return_tensors="pt").to(device)
    print("tokenizer", time.time() - start_at)

    start_at = time.time()
    # Generate the output
    outputs = model.generate(**inputs, max_length=max_length)
    print("generate", time.time() - start_at)

    start_at = time.time()
    # Decode and print the output
    s = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(code):]
    print("decode", time.time() - start_at)
    start_at = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    print("gc", time.time() - start_at)
    return s