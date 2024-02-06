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
pip install -r requirements.txt
```

### Training the Model
Run the script to start the training process. If a pre-trained model is not found, the script automatically initiates training:

```bash
python main.py
```

### Generating Text
After training or loading an existing model, you can generate text based on prompts by following the interactive command prompt instructions.


## Tips and tricks to cut training, validation and test time

### Model suggestions

`distilgpt2` is a distilled version of GPT-2 that retains much of the original's capabilities but is smaller and faster to train:

- distilgpt2
- gpt2

### Dataset suggestions

- wikitext-2-v1 (44.8k rows)
- wikitext-103-v1 (1.81 M rows)

### Reduce Dataset Size

Another way to reduce training time is by using a smaller portion of your dataset. Instead of training on the entire dataset, you can sample a subset of your data for training. This will reduce the amount of computation needed per epoch:

```python
from torch.utils.data import Subset
import numpy as np

# Assuming `train_dataset` is your full training dataset
indices = np.random.choice(len(train_dataset), size=int(len(train_dataset) * 0.1), replace=False) # Sample 10% of the dataset
small_train_dataset = Subset(train_dataset, indices)
```
Then, use small_train_dataset in your DataLoader instead of the full dataset.

### Decrease the Number of Epochs

Reducing the number of epochs will directly decrease training time. If you're just looking to learn the concepts, even a single epoch can be insightful:

```python
num_epochs = 1  # Reduce the number of epochs to 1 or a few

```

###  Increase the Learning Rate

Sometimes, increasing the learning rate can help the model converge faster. However, this needs to be done carefully to avoid overshooting the minimum of the loss function. It's more of a trial-and-error process:

```python
optimizer = AdamW(model.parameters(), lr=1e-4)  # Adjust the learning rate as needed
```

### Utilize Gradient Accumulation

If you're limited by the GPU memory and hence using a very small batch size, you can implement gradient accumulation to effectively increase the batch size without increasing the memory requirement. This method involves summing the gradients over several iterations and only updating the model weights after a specified number of steps. This can sometimes make training more efficient.





### Test questions for the wikitext-2-v1 dataset

- "In 2005, the Nobel Prize in Physics was awarded to John L. Hall and Theodor W. Hänsch for their work on precision spectroscopy and optical frequency combs."

- "The Renaissance, a period of great cultural and artistic growth in Europe, began in the 14th century and continued into the 17th century."

- "Marie Curie, a pioneering physicist and chemist, was the first woman to win a Nobel Prize and remains one of the most influential scientists in history."

- "The Industrial Revolution, a transformative era in human history, saw the rapid development of machinery, factories, and urbanization."

- "The theory of relativity, developed by Albert Einstein in the early 20th century, revolutionized our understanding of space, time, and gravity."