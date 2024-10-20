import json
from transformers import PreTrainedTokenizerFast ,LlamaForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import os
from getpass import getpass

def get_access_token():
    token = os.environ.get('HUGGING_FACE_TOKEN')
    if not token:
        token = getpass("Enter your Hugging Face access token: ")
    return token

# Get the access token
access_token = get_access_token()

# Print debug information
print(f"Access token type: {(access_token)}")
print(f"Access token length: {len(access_token)}")

# Load the pre-trained LLaMA model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-70B"
print(f"Model name: {model_name}")

try:
    print("Attempting to load tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, token=access_token, legacy=False)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    raise

try:
    print("Attempting to load model...")
    model = LlamaForCausalLM.from_pretrained(model_name, token=access_token)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Rest of the code remains the same
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Assume your JSON file is named 'custom_bob.json'
dataset = load_dataset('custom_bob.json')

def preprocess_function(examples):
    conversations = [f"User: {item['user']}\nAssistant: {item['assistant']}" for item in examples]
    return tokenizer(conversations, truncation=True, padding="max_length", max_length=512)

hf_dataset = Dataset.from_list(dataset)
tokenized_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=hf_dataset.column_names)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Fine-tuning complete. Model saved to './fine_tuned_llama'")