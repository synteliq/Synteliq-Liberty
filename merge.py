from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, merge_and_unload
from huggingface_hub import login
import torch

# Step 1: Login to Hugging Face
login("your_hf_token_here")  # 🔑 Replace with your Hugging Face token

# Step 2: Adapter model ID
peft_model_id = "Synteliq/Liberty_v1"

# Step 3: Load PEFT config to get base model info
config = PeftConfig.from_pretrained(peft_model_id)

# Step 4: Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Step 5: Load adapter (LoRA)
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Step 6: Merge adapter into base model
merged_model = merge_and_unload(model)

# Step 7: Push merged model to Hugging Face under new repo name
merged_model.push_to_hub("Synteliq/Liberty-Base", use_auth_token=True)

# Step 8: Push tokenizer to the same repo
AutoTokenizer.from_pretrained(config.base_model_name_or_path).push_to_hub("Synteliq/Liberty-Base", use_auth_token=True)

print("✅ Merged model successfully uploaded to Hugging Face!")
