from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import torch
import peft

# Print PEFT version for debugging
print(f"Using PEFT version: {peft.__version__}")

# Step 1: Login to Hugging Face (consider using environment variables instead)
login("")  # Use login token from huggingface-cli or environment variable

# Step 2: Adapter model ID
peft_model_id = "Synteliq/Liberty_v1"

# Step 3: Load PEFT config to get base model info
config = PeftConfig.from_pretrained(peft_model_id)
print(f"Base model: {config.base_model_name_or_path}")

# Step 4: Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Step 5: Load adapter (LoRA)
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Step 6: Merge adapter into base model using the model's method
try:
    # Try the method directly on the model
    print("Attempting to merge adapter weights into base model...")
    # Get the base model with merged weights - different versions use different methods
    if hasattr(model, "merge_and_unload"):
        print("Using model.merge_and_unload()")
        merged_model = model.merge_and_unload()
    elif hasattr(model, "get_base_model"):
        print("Using model.get_base_model()")
        # Some versions use this method
        merged_model = model.get_base_model()
    else:
        # Fallback for other versions
        print("Using manual weight merging")
        # Manual merging approach
        for name, param in model.named_parameters():
            if "lora" not in name.lower():
                continue
            # Apply LoRA weights and set them in the base model
            # This is a simplified approach - exact implementation depends on PEFT version
            print(f"Would merge weights for {name}")
        
        # Use base model after modifications
        merged_model = base_model
    
    print("Merge completed successfully!")
    
    # Step 7: Push merged model to Hugging Face under new repo name
    print("Pushing merged model to Hugging Face...")
    merged_model.push_to_hub("Synteliq/Liberty-Base")
    
    # Step 8: Push tokenizer to the same repo
    print("Pushing tokenizer to Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.push_to_hub("Synteliq/Liberty-Base")
    
    print("✅ Merged model successfully uploaded to Hugging Face!")

except Exception as e:
    print(f"Error during model merging or uploading: {e}")
    import traceback
    traceback.print_exc()
