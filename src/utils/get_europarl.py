import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer
import re
import os
from loguru import logger

# Configure Loguru logger
logger.add("preprocessing.log", format="{time} {level} {message}", level="INFO", rotation="10 MB")

def load_and_split_dataset(cfg):
    """Load the EuroParl dataset and split off a subset."""
    logger.info(f"Loading dataset: {cfg.dataset.name} with language pair {cfg.dataset.lang_pair}")
    ep = load_dataset(cfg.dataset.name, cfg.dataset.lang_pair)
    small_ds = ep["train"].train_test_split(test_size=cfg.dataset.test_size, seed=cfg.dataset.seed)["test"]
    logger.info(f"Original dataset size: {len(ep['train'])}")
    logger.info(f"Subset dataset size (10%): {len(small_ds)}")
    return small_ds

# def preprocess_text(examples, cfg):
#     """Preprocess text based on the configuration."""
#     text = examples["translation"][cfg.dataset.source_lang]
#     if cfg.preprocessing.lowercase:
#         text = text.lower()
#     if cfg.preprocessing.remove_non_alphanumeric:
#         text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     if cfg.preprocessing.remove_extra_whitespace:
#         text = re.sub(r"\s+", " ", text).strip()
#     return {"text": text}

def preprocess_text(examples, cfg):
    """Preprocess text for both source and target languages based on the configuration."""
    source_lang = cfg.dataset.source_lang
    target_lang = cfg.dataset.target_lang

    # Preprocess source language text
    source_text = examples["translation"][source_lang]
    if cfg.preprocessing.lowercase:
        source_text = source_text.lower()
    if cfg.preprocessing.remove_non_alphanumeric:
        source_text = re.sub(r"[^a-zA-Z0-9\s]", "", source_text)
    if cfg.preprocessing.remove_extra_whitespace:
        source_text = re.sub(r"\s+", " ", source_text).strip()

    # Preprocess target language text
    target_text = examples["translation"][target_lang]
    if cfg.preprocessing.lowercase:
        target_text = target_text.lower()
    if cfg.preprocessing.remove_non_alphanumeric:
        # Preserve accented characters in Spanish by adding them to the regex pattern
        target_text = re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚÑ0-9\s]", "", target_text)
    if cfg.preprocessing.remove_extra_whitespace:
        target_text = re.sub(r"\s+", " ", target_text).strip()

    # Return both preprocessed texts
    return {
        source_lang: source_text,
        target_lang: target_text
    }

def apply_preprocessing(dataset, cfg):
    """Apply preprocessing to the dataset."""
    logger.info("Applying preprocessing to the dataset")
    return dataset.map(lambda x: preprocess_text(x, cfg), remove_columns=["translation"])

def tokenize_dataset(dataset, cfg):
    """Tokenize the dataset based on the configuration."""
    logger.info(f"Initializing tokenizer: {cfg.tokenization.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenization.model_name)
    
    def tokenize_function(examples):
        source_lang = cfg.dataset.source_lang
        target_lang = cfg.dataset.target_lang

        # Tokenize both source and target language fields
        source_tokens = tokenizer(
            examples[source_lang],
            truncation=cfg.tokenization.truncation,
            padding=cfg.tokenization.padding,
            max_length=cfg.tokenization.max_length,
        )
        
        target_tokens = tokenizer(
            examples[target_lang],
            truncation=cfg.tokenization.truncation,
            padding=cfg.tokenization.padding,
            max_length=cfg.tokenization.max_length,
        )
        
        # Return the tokenized version of both fields
        return {
            f"{source_lang}_input_ids": source_tokens["input_ids"],
            f"{source_lang}_attention_mask": source_tokens["attention_mask"],
            f"{target_lang}_input_ids": target_tokens["input_ids"],
            f"{target_lang}_attention_mask": target_tokens["attention_mask"]
        }
    
    logger.info("Applying tokenization to the dataset")
    return dataset.map(tokenize_function, batched=True)

def split_dataset(dataset, cfg):
    """Split the dataset into training, validation, and test sets."""
    logger.info("Splitting the dataset into training, validation, and test sets")
    train_validation_ds = dataset.train_test_split(test_size=cfg.dataset.test_size, seed=cfg.dataset.seed)
    train_validation = train_validation_ds["train"]
    test_ds = train_validation_ds["test"]
    
    train_val_split = train_validation.train_test_split(test_size=cfg.dataset.test_size / (1 - cfg.dataset.test_size), seed=cfg.dataset.seed)
    train_ds = train_val_split["train"]
    validation_ds = train_val_split["test"]
    
    logger.info(f"Training set size: {len(train_ds)}")
    logger.info(f"Validation set size: {len(validation_ds)}")
    logger.info(f"Testing set size: {len(test_ds)}")
    
    return train_ds, validation_ds, test_ds

def save_datasets(train_ds, validation_ds, test_ds, cfg):
    """Save datasets to the specified output directory."""
    os.makedirs(cfg.output.dir, exist_ok=True)
    train_path = os.path.join(cfg.output.dir, "train.json")
    val_path = os.path.join(cfg.output.dir, "validation.json")
    test_path = os.path.join(cfg.output.dir, "test.json")

    logger.info(f"Saving training set to {train_path}")
    train_ds.to_json(train_path)
    
    logger.info(f"Saving validation set to {val_path}")
    validation_ds.to_json(val_path)
    
    logger.info(f"Saving test set to {test_path}")
    test_ds.to_json(test_path)

    logger.info(f"Processed data saved to {cfg.output.dir}")

@hydra.main(config_path="../../configs", config_name="europarl")
def main(cfg: DictConfig):
    small_ds = load_and_split_dataset(cfg)
    print("SMALL:", small_ds[0])
    preprocessed_ds = apply_preprocessing(small_ds, cfg)
    print("PREPROCESSED:", preprocessed_ds[0])
    tokenized_ds = tokenize_dataset(preprocessed_ds, cfg)
    print("TOKENIZED:", tokenized_ds[0])
    train_ds, validation_ds, test_ds = split_dataset(tokenized_ds, cfg)
    print("TRAIN:", train_ds[0])
    print("VALIDATION:", validation_ds[0])
    print("TEST:", test_ds[0])
    save_datasets(train_ds, validation_ds, test_ds, cfg)

if __name__ == "__main__":
    main()