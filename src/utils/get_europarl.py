from datasets import load_dataset
from transformers import AutoTokenizer
import re

# Load EuroParl dataset (e.g., English-Spanish for multilingual translation tasks)
from dataloader import ep


# Split off 10% of the dataset
small_ds = ep["train"].train_test_split(test_size=0.1, seed=42)["test"]
print(f"Original dataset size: {len(ep['train'])}")
print(f"Subset dataset size (10%): {len(small_ds)}")


def preprocess_text(examples):
    text = examples["translation"]["en"]  # Or change to "es" for Spanish
    # Lowercasing
    text = text.lower()
    # Removing non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Removing extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return {"text": text}

# Apply preprocessing
small_ds = small_ds.map(preprocess_text, remove_columns=["translation"])

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Define a function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization
tokenized_small_ds = small_ds.map(tokenize_function, batched=True)

# First, we split the 10% subset into 90% train+validation and 10% test
train_validation_ds = small_ds.train_test_split(test_size=0.1, seed=42)
train_validation = train_validation_ds["train"]
test_ds = train_validation_ds["test"]

# Next, we split the 90% train+validation set into 80% train and 10% validation (relative to the original data)
train_val_split = train_validation.train_test_split(test_size=1/9, seed=42)
train_ds = train_val_split["train"]
validation_ds = train_val_split["test"]

# Print the sizes to confirm
print(f"Training set size: {len(train_ds)}")
print(f"Validation set size: {len(validation_ds)}")
print(f"Testing set size: {len(test_ds)}")
