from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel, PeftConfig
import torch

peft_model_id = "Synteliq/Liberty_v1"
config = PeftConfig.from_pretrained(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda")  # Send manually to GPU

model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.to("cuda").eval()

streamer = TextStreamer(tokenizer)

print("Model loaded on GPU! Type something (or 'exit'):")
while True:
    prompt = input("\nUser: ")
    if prompt.lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        streamer=streamer
    )

