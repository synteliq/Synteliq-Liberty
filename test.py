import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """
    Loads the fine-tuned model and tokenizer from the given path.
    """
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generates text from the fine-tuned model based on the user's input prompt.
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate text from the model
    output = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)

    # Decode the output and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def evaluate_model(model, tokenizer):
    """
    Evaluates the fine-tuned model with user input and generates a response.
    """
    while True:
        # Get the user's input prompt
        prompt = input("\nEnter a prompt (or 'exit' to quit): ")

        if prompt.lower() == 'exit':
            print("Exiting the program.")
            break
        
        # Generate text based on the user input
        generated_text = generate_text(model, tokenizer, prompt)
        print("\nGenerated Response:\n", generated_text)

def main():
    model_path = "./synteliq-lora"  # Path to your fine-tuned model directory
    model, tokenizer = load_model(model_path)

    print("Model loaded successfully!\n")
    evaluate_model(model, tokenizer)

if __name__ == "__main__":
    main()
