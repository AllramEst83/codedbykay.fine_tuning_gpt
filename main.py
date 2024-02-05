from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
import os

# Constants for model and tokenizer paths
MODEL_PATH = "./model/gpt2_finetuned_model"
TOKENIZER_PATH = "./model/gpt2_finetuned_tokenizer"

# Ensure CUDA is available and select the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer globally to ensure it's accessible throughout the script
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize model variable globally
model = None

class WikiTextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long)
        }
        return item

def tokenize_function(examples):
    # Use the globally defined tokenizer
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def train_model():
    global model  # Declare model as global to modify the global instance
    print("Downloading and loading the dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    print("Dataset successfully downloaded and loaded.")
    
    print("Tokenizing the dataset. This might take a while...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("Tokenization complete.")

    train_dataset = WikiTextDataset(tokenized_datasets['train'])
    print("Dataset prepared for training.")

    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print("DataLoader initialized.")

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    print("Model loaded and sent to the device.")

    optimizer = AdamW(model.parameters(), lr=5e-5)
    print("Optimizer prepared.")

    model.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started.")
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print("Model and tokenizer saved.")

def check_and_load_model():
    global model, tokenizer  # Declare as global to modify the global instance
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print(f"A trained model already exists. Loading model and tokenizer.")
        tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.to(device)
        print("Loaded model and tokenizer from saved files.")
    else:
        print(f"A trained model could not be found, Initiating training of GPT-2 model.")
        train_model()

def generate_text(prompt, max_length=50, temperature=1.0, num_return_sequences=1):
    global model  # Ensure model is recognized within this function
    print(f"Generating text for prompt: '{prompt}'")
    model.eval()  # Set model to evaluation mode

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    for i, generated_sequence in enumerate(output_sequences):
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        print(f"Generated Text {i+1}: {text}")

def main():
    choice = input("Enter your choice (1: Load Model, 2: Generate Text): ")
    if choice == "1":
        check_and_load_model()
    elif choice == "2":
        prompt = input("Enter the prompt: ")
        generate_text(prompt, max_length=50, temperature=0.7, num_return_sequences=3)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
