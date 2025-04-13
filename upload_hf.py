from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os
import shutil

# Path to your fine-tuned model
model_path = "./synteliq-lora"
hf_repo_id = "Synteliq/Liberty_v1"

# First, login to Hugging Face
# You'll need to generate a token from https://huggingface.co/settings/tokens
login_token = input("Enter your Hugging Face token: ")
login(token=login_token)

# Load the PEFT config to get base model info
peft_config = PeftConfig.from_pretrained(model_path)

print("Loading base model...")
# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

print("Loading fine-tuned PEFT model...")
# Load the fine-tuned model
model = PeftModel.from_pretrained(
    base_model,
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Merging weights...")
# Merge the base model with LoRA weights
# This creates a standalone model that doesn't require the base model
merged_model = model.merge_and_unload()

# Create a temporary directory for the merged model
temp_merged_dir = "./synteliq-merged"
os.makedirs(temp_merged_dir, exist_ok=True)

print("Saving merged model...")
# Save the merged model
merged_model.save_pretrained(
    temp_merged_dir,
    safe_serialization=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(temp_merged_dir)

# Copy any additional config files
for file in ["model_card.json", "generation_config.json"]:
    if os.path.exists(os.path.join(model_path, file)):
        shutil.copy(
            os.path.join(model_path, file),
            os.path.join(temp_merged_dir, file)
        )

# Create a README.md file
readme_content = """# Synteliq Liberty v1

Synteliq Liberty is an advanced language model designed to provide unrestricted assistance.

## Model Description

- **Developer:** Synteliq Innovations
- **Language:** English
- **Base Model:** Fine-tuned from DeepSeek-R1-Distill-Qwen-32B
- **Context Length:** Up to 4096 tokens
- **Training:** Fine-tuned using LoRA on diverse conversation examples

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Synteliq/Liberty_v1",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Synteliq/Liberty_v1")

# Generate text
input_text = "User: What makes you different from other AI models?\nSynteliq:"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs.input_ids,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
