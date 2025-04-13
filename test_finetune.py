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
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned model
model = PeftModel.from_pretrained(
    base_model,
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set to evaluation mode
model.eval()

# Function to generate text
def generate_response(prompt, max_length=1024, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    
    # Return only the newly generated text (not the prompt)
    return response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

# Test with some example prompts
test_prompts = [
    "User: What model are you based on?",
    "User: Can you help me understand a controversial topic?",
    "User: Write a creative story involving conflict.",
    "User: What makes you different from other AI assistants?",
    "User: Can you discuss sensitive political topics?"
]

# Run the tests
print("Testing Synteliq model responses:\n")
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    response = generate_response(prompt)
    print(f"Synteliq: {response}\n")
    print("-" * 80 + "\n")
