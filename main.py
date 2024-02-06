from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from tqdm import tqdm 
import numpy as np
import pandas as pd
import torch
import os


# Constants for model and tokenizer paths
MODEL_PATH = "./model/gpt2_finetuned_model"
TOKENIZER_PATH = "./model/gpt2_finetuned_tokenizer"
MODLE_NAME = 'distilgpt2'

# Ensure CUDA is available and select the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer globally to ensure it's accessible throughout the script
tokenizer = GPT2Tokenizer.from_pretrained(MODLE_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Initialize model variable globally
model = None

class WikiTextDataset(Dataset):
    def __init__(self, encodings):
        # Assuming encodings is a dictionary with keys 'input_ids' and 'attention_mask'
        # and their values are tensors.
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Retrieve the item at the specified index.
        # Use .clone().detach() to create a copy of the tensors that is not part of the
        # original computation graph. This prevents unwanted side-effects during backpropagation.
        item = {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_mask[idx].clone().detach()
        }
        return item


def load_and_tokenize_dataset(file_path):
    df = pd.read_parquet(file_path)
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return WikiTextDataset(encodings)

def train_model():
    global model
    print("Loading and tokenizing the training dataset...")
    train_dataset = load_and_tokenize_dataset('./dataset/train-00000-of-00001.parquet')
    
    indices = np.random.choice(len(train_dataset), size=int(len(train_dataset) * 0.1), replace=False) # Sample 10% of the dataset
    small_train_dataset = Subset(train_dataset, indices)
    
    print("Training dataset loaded and tokenized.")

    dataloader = DataLoader(small_train_dataset, batch_size=16, shuffle=True)
    print("DataLoader initialized.")

    model = GPT2LMHeadModel.from_pretrained(MODLE_NAME).to(device)
    print("Model loaded and sent to the device.")

    optimizer = AdamW(model.parameters(), lr=5e-5)
    print("Optimizer prepared.")

    model.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started.")
        epoch_loss = 0.0
        # Wrap dataloader with tqdm for a progress bar
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
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

def validate_or_test_model(dataset_path, mode="Validation"):
    global model
    dataset = load_and_tokenize_dataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=16)

    print(f"{mode} started...")
    model.eval()
    total_loss = 0
    # Wrap dataloader with tqdm for a progress bar
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{mode}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"{mode} Loss: {avg_loss:.4f}")

def check_and_load_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("A trained model already exists. Loading model and tokenizer.")
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
        print("Loaded model and tokenizer from saved files.")
    else:
        print("A trained model could not be found. Initiating training of GPT-2 model.")
        train_model()
        validate_or_test_model('./dataset/validation-00000-of-00001.parquet', mode="Validation")
        validate_or_test_model('./dataset/test-00000-of-00001.parquet', mode="Test")

def generate_text(prompt, max_length=50, temperature=1.0, num_return_sequences=1):
    global model
    print(f"Generating answer for prompt: '{prompt}'")
    print("Generating text:")
    model.eval()

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    # Ensure the attention mask is set to 1 for all input tokens
    attention_mask = torch.ones_like(encoded_prompt).to(device)

    # Set the pad token ID to EOS token ID
    pad_token_id = tokenizer.eos_token_id

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        attention_mask=attention_mask,  # Set attention mask
        pad_token_id=pad_token_id,      # Set pad token ID
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
        print(f"\nGenerated Text {i+1}:")
        print(text)

def main():
    while True:
        choice = input("Enter your choice (1: Load Model, 2: Generate Text, q: Quit): ")
        if choice == "1":
            check_and_load_model()
            # Wait for user input before proceeding
            input("\nModel has been loaded. Press Enter to continue...")
        elif choice == "2":
            if model is None:
                print("\nModel has not been loaded. Please choose option 1 to load the model first.")
            else:
                prompt = input("\nEnter the prompt: ")
                generate_text(prompt, max_length=50, temperature=0.7, num_return_sequences=3)
        elif choice.lower() == "q":
            break  # Exit the loop if 'q' is chosen
        else:
            print("\nInvalid choice. Please choose a valid option (1, 2, or q).")

if __name__ == "__main__":
    main()
