import re
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MBartTokenizer
from datasets import load_dataset

class CorpusDataset:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_dataset(self):
        """Load the dataset from Hugging Face."""
        logger.info(f"Loading dataset: {self.cfg.dataset.name}")
        return load_dataset(self.cfg.dataset.name, self.cfg.dataset.language_pair)

    def preprocess_text(self, examples):
        """Preprocess source and target texts based on configuration."""
        source = self.cfg.dataset.source
        target = self.cfg.dataset.target

        # Access the nested 'translation' field for both source and target languages
        input_text = examples["translation"].get(source, "")
        label_text = examples["translation"].get(target, "")

        if self.cfg.preprocessing.lowercase:
            input_text = input_text.lower()
            label_text = label_text.lower()

        if self.cfg.preprocessing.remove_non_alphanumeric:
            input_text = re.sub(r"[^a-zA-Z0-9\s]", "", input_text)
            label_text = re.sub(r"[^a-zA-Z0-9\s]", "", label_text)

        if self.cfg.preprocessing.remove_extra_whitespace:
            input_text = re.sub(r"\s+", " ", input_text).strip()
            label_text = re.sub(r"\s+", " ", label_text).strip()

        return {
            "input_text": input_text,
            "label_text": label_text
        }

    def tokenize(self, dataset):
        """Tokenize the dataset using the specified tokenizer."""
        if self.cfg.tokenization.model_name == "facebook/mbart-large-50-many-to-many-mmt":
            tokenizer = MBartTokenizer.from_pretrained(self.cfg.tokenization.model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenization.model_name)
        
        max_length = self.cfg.tokenization.max_length
        logger.info("Tokenizing the dataset ...")

        def tokenize_function(examples):
            input_text = examples["input_text"]
            label_text = examples["label_text"]

            # Tokenize input and label as single sequences
            input_tokenized = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=None,  
                padding=False     
            )
            
            label_tokenized = tokenizer(
                label_text,
                return_tensors="pt",
                max_length=None,  
                padding=False
            )

            return {
                "input_ids": input_tokenized["input_ids"][0],  
                "attention_mask": input_tokenized["attention_mask"][0],
                "labels": label_tokenized["input_ids"][0]  
            }

        return dataset.map(tokenize_function, batched=False)


    def split_dataset(self, dataset):
        """Split the dataset into train, validation, and test sets."""
        logger.info("Splitting the dataset into training, validation, and test sets")

        train_split = dataset["train"]

        # Create the test split
        train_test_split = train_split.train_test_split(test_size=self.cfg.dataset.test_size, seed=self.cfg.dataset.seed)
        test_ds = train_test_split['test']
        remaining_train = train_test_split['train']

        # Create the validation split from the remaining training data
        val_size = self.cfg.dataset.val_size / (1 - self.cfg.dataset.test_size)  # Adjust for the remaining data
        train_val_split = remaining_train.train_test_split(test_size=val_size, seed=self.cfg.dataset.seed)
        train_ds = train_val_split['train']
        val_ds = train_val_split['test']

        logger.info(f"Training set size: {len(train_ds)}")
        logger.info(f"Validation set size: {len(val_ds)}")
        logger.info(f"Test set size: {len(test_ds)}")

        return train_ds, val_ds, test_ds

    def process_and_save(self):
        """Complete pipeline: load, preprocess, tokenize, split, and save the dataset."""
        dataset = self.load_dataset()
        preprocessed_dataset = dataset.map(self.preprocess_text, remove_columns=["translation"])
        tokenized_dataset = self.tokenize(preprocessed_dataset)
        train_ds, val_ds, test_ds = self.split_dataset(tokenized_dataset)
        return train_ds, val_ds, test_ds
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-length input for DataLoader."""
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch], 
            batch_first=True, 
            padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch], 
            batch_first=True, 
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item["labels"], dtype=torch.long) for item in batch], 
            batch_first=True, 
            padding_value=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class EuroParlDataset(CorpusDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_dataset(self):
        logger.info(f"Loading EuroParl dataset: {self.cfg.dataset.name} with language pair {self.cfg.dataset.language_pair}")
        return load_dataset(self.cfg.dataset.name, self.cfg.dataset.language_pair)
    
class TedTalksDataset(CorpusDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_dataset(self):
        """Load the dataset from Hugging Face with trust_remote_code enabled."""
        logger.info(f"Loading dataset: {self.cfg.dataset.name}")

        if 'language_pair' in self.cfg.dataset and 'year' in self.cfg.dataset:
            return load_dataset(
                self.cfg.dataset.name,
                language_pair=tuple(self.cfg.dataset.language_pair),
                year=self.cfg.dataset.year,
                trust_remote_code=True  
            )
        elif 'language_pair' in self.cfg.dataset:
            return load_dataset(
                self.cfg.dataset.name,
                language_pair=tuple(self.cfg.dataset.language_pair),
                trust_remote_code=True  
            )
        else:
            return load_dataset(self.cfg.dataset.name, trust_remote_code=True) 
    
class MultiLexSumDataset(CorpusDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_dataset(self):
        logger.info(f"Loading Multi-LexSum dataset: {self.cfg.dataset.name} with config version {self.cfg.dataset.config_version}")
        return load_dataset(self.cfg.dataset.name, self.cfg.dataset.config_version, trust_remote_code=True)

    def preprocess_text(self, examples):
        """Preprocess text for source and target with selected summary length."""
        input_text = " ".join(examples[self.cfg.dataset.source_field])  
        summary_length = self.cfg.dataset.target_summary_length
        label_text = examples[f"summary/{summary_length}"]  

        if self.cfg.preprocessing.lowercase:
            input_text = input_text.lower()
            label_text = label_text.lower()
        if self.cfg.preprocessing.remove_non_alphanumeric:
            input_text = re.sub(r"[^a-zA-Z0-9\s]", "", input_text)
            label_text = re.sub(r"[^a-zA-Z0-9\s]", "", label_text)
        if self.cfg.preprocessing.remove_extra_whitespace:
            input_text = re.sub(r"\s+", " ", input_text).strip()
            label_text = re.sub(r"\s+", " ", label_text).strip()

        return {
            "input_text": input_text,
            "label_text": label_text
        }
    
    def process_and_save(self):
        """Complete pipeline: load, preprocess, tokenize, split, and save the dataset."""
        dataset = self.load_dataset()
        preprocessed_dataset = dataset.map(self.preprocess_text)
        tokenized_dataset = self.tokenize(preprocessed_dataset)
        train_ds, val_ds, test_ds = self.split_dataset(tokenized_dataset)
        return train_ds, val_ds, test_ds


def get_dataloader(cfg, split="train", batch_size=16, shuffle=True, num_workers=4):
    """
    Function to create a DataLoader for the correct dataset class based on the configuration.
    
    Args:
        cfg: Configuration object.
        split (str): Which split to load ('train', 'val', or 'test').
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        
    Returns:
        DataLoader: A DataLoader for the specified dataset split.
    """
    if cfg.dataset.name == "Helsinki-NLP/europarl":
        dataset = EuroParlDataset(cfg)
    elif cfg.dataset.name == "IWSLT/ted_talks_iwslt":
        dataset = TedTalksDataset(cfg)
    elif cfg.dataset.name == "allenai/multi_lexsum":
        dataset = MultiLexSumDataset(cfg)
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.dataset.name}")
    
    train_ds, val_ds, test_ds = dataset.process_and_save()
    
    if split == "train":
        selected_dataset = train_ds
    elif split == "val":
        selected_dataset = val_ds
    elif split == "test" and test_ds is not None:
        selected_dataset = test_ds
    else:
        raise ValueError(f"Invalid split '{split}' specified or test set not available.")
    
    dataloader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers
    )
    
    return dataloader

