from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the saved model
output_dir = "./synteliq-lora"  # Update if necessary

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Set the model to evaluation mode
model.eval()

# Testing input
input_text = "Your input text here"  # Provide the input you want to test the model with

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Run the model (ensure you're running it on the right device)
with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to(model.device), max_length=200, num_return_sequences=1)

# Decode the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Output: ", output_text)
