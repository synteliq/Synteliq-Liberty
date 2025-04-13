from huggingface_hub import Repository, HfApi

# Define your local model directory and the Hugging Face repository URL
model_dir = "./synteliq-lora"  # Your model's local directory path
repo_name = "synteliq-liberty_v1"  # Your model name on Hugging Face
repo_url = f"your_username/{repo_name}"  # Replace with your Hugging Face repo URL

# Initialize Hugging Face API and create a repo if needed
api = HfApi()

# Create a new model repository if it doesn't exist
try:
    api.create_repo(repo_url, exist_ok=True)
    print(f"Repository '{repo_name}' created or already exists.")
except Exception as e:
    print(f"Error creating repo: {e}")

# Clone the Hugging Face repository to your local machine
repo = Repository(local_dir=model_dir, clone_from=repo_url)

# Upload the model to Hugging Face Hub
repo.push_to_hub(commit_message="Add synteliq-liberty_v1 model files")

# Optionally, you can push additional files like tokenizer and config (if required)
# You can upload tokenizer files as follows:
# tokenizer.push_to_hub(repo_url)
