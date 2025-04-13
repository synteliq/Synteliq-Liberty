from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Path to your fine-tuned model
model_path = "./synteliq-lora"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the PEFT configuration
peft_config = PeftConfig.from_pretrained(model_path)

# Load the base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned model
print("Loading fine-tuned model...")
model = PeftModel.from_pretrained(
    base_model,
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set to evaluation mode
model.eval()

def generate_response(prompt, conversation_history="", max_length=2048, temperature=0.7):
    # Combine history with new prompt
    full_prompt = conversation_history + "User: " + prompt + "\nSynteliq:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's response by finding where "Synteliq:" appears last
    response_start = response.rfind("Synteliq:") + len("Synteliq:")
    model_response = response[response_start:].strip()
    
    # Update conversation history
    updated_history = full_prompt + " " + model_response + "\n\n"
    
    return model_response, updated_history

print("\n===== Synteliq Interactive Chat =====")
print("Type 'exit' to end the conversation.\n")

conversation_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Ending conversation.")
        break
    
    response, conversation_history = generate_response(user_input, conversation_history)
    print(f"Synteliq: {response}\n")
