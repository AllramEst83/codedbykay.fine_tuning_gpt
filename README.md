# GPT-2 Fine-Tuning and Text Generation

This project fine-tunes the GPT-2 model from Hugging Face's Transformers library on the WikiText-103 dataset. It includes functionality for training the model, saving the trained model and tokenizer, loading them for future use, and generating text based on input prompts.

## Getting Started

### Dependencies

- Python 3.8+
- PyTorch
- Hugging Face's Transformers
- Datasets library

### Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
pip install torch transformers datasets
```

### Training the Model
Run the script to start the training process. If a pre-trained model is not found, the script automatically initiates training:

```bash
python main.py
```

### Generating Text
After training or loading an existing model, you can generate text based on prompts by following the interactive command prompt instructions.
