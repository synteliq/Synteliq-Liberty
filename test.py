from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./synteliq-lora"  # Path to the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to the appropriate device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate text function
def generate_text(prompt, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text using the model
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example prompt for testing
prompt = "Once upon a time"
generated_text = generate_text(prompt)

# Print the generated text
print("Generated Text:")
print(generated_text)
