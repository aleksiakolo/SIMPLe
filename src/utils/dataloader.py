import re
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class CorpusDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pad_token_id = 0  # overwritten once the tokenizer is loaded

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

        return {
            "input_text": self.clean_text(input_text),
            "label_text": self.clean_text(label_text)
        }

    def clean_text(self, text):
        """Apply the configured normalisation. Pretrained seq2seq models expect cased text with
        punctuation, so lowercasing and character stripping should normally stay off."""
        if self.cfg.preprocessing.lowercase:
            text = text.lower()
        if self.cfg.preprocessing.remove_non_alphanumeric:
            # \w is Unicode-aware, so accented letters (á, ñ, ü, ...) are kept
            text = re.sub(r"[^\w\s]", "", text)
        if self.cfg.preprocessing.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, dataset):
        """Tokenize the dataset using the specified tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenization.model_name)
        if hasattr(tokenizer, "lang_code_to_id"):
            # mBART-50 needs language codes (e.g. en_XX, es_XX) prepended to source and target
            tokenizer.src_lang = self.mbart_lang_code(tokenizer, self.cfg.dataset.source)
            tokenizer.tgt_lang = self.mbart_lang_code(tokenizer, self.cfg.dataset.target)
        self.pad_token_id = tokenizer.pad_token_id

        max_length = self.cfg.tokenization.max_length
        logger.info("Tokenizing the dataset ...")

        def tokenize_function(examples):
            input_text = examples["input_text"]
            label_text = examples["label_text"]

            # Tokenize input and label as single sequences
            input_tokenized = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,  
                truncation=True,
                padding=False     
            )
            
            label_tokenized = tokenizer(
                text_target=label_text,
                return_tensors="pt",
                max_length=max_length,  
                truncation=True,
                padding=False
            )

            return {
                "input_ids": input_tokenized["input_ids"][0],  
                "attention_mask": input_tokenized["attention_mask"][0],
                "labels": label_tokenized["input_ids"][0]  
            }

        return dataset.map(tokenize_function, batched=False)


    def split_dataset(self, dataset):
        """Return (train, val, test) splits.

        The dataset's own validation/test splits are used when they exist; otherwise they are
        carved out of the training split using test_size and val_size (fractions of the full set).
        An optional subset_fraction is applied once to each split to reduce compute.
        """
        logger.info("Splitting the dataset into training, validation, and test sets")
        seed = self.cfg.dataset.seed
        train_ds = dataset["train"]

        if "test" in dataset:
            test_ds = dataset["test"]
        else:
            split = train_ds.train_test_split(test_size=self.cfg.dataset.test_size, seed=seed)
            train_ds, test_ds = split["train"], split["test"]

        if "validation" in dataset:
            val_ds = dataset["validation"]
        else:
            # val_size is a fraction of the full dataset, so rescale it to what remains after the test split
            remaining = 1 - (0 if "test" in dataset else self.cfg.dataset.test_size)
            split = train_ds.train_test_split(test_size=self.cfg.dataset.val_size / remaining, seed=seed)
            train_ds, val_ds = split["train"], split["test"]

        fraction = self.cfg.dataset.get("subset_fraction", 1.0)
        if fraction < 1.0:
            train_ds, val_ds, test_ds = (
                ds.shuffle(seed=seed).select(range(max(1, int(fraction * len(ds)))))
                for ds in (train_ds, val_ds, test_ds)
            )

        logger.info(f"Training set size: {len(train_ds)}")
        logger.info(f"Validation set size: {len(val_ds)}")
        logger.info(f"Test set size: {len(test_ds)}")

        return train_ds, val_ds, test_ds

    def prepare_split(self, split_ds):
        """Preprocess and tokenize a single split, dropping the raw columns."""
        preprocessed = split_ds.map(self.preprocess_text, remove_columns=split_ds.column_names)
        return self.tokenize(preprocessed)

    def process_and_save(self):
        """Complete pipeline: load, split, then preprocess and tokenize each split."""
        dataset = self.load_dataset()
        splits = self.split_dataset(dataset)
        return tuple(self.prepare_split(ds) for ds in splits)

    @staticmethod
    def mbart_lang_code(tokenizer, lang):
        """Map an ISO language code like 'es' to the mBART-50 code 'es_XX'."""
        for code in tokenizer.lang_code_to_id:
            if code.split("_")[0] == lang:
                return code
        raise ValueError(f"Language '{lang}' is not supported by mBART-50")

    def collate_fn(self, batch):
        """Pad variable-length examples. Labels are padded with -100 so padding is ignored by the loss."""
        def pad(key, value):
            return torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item[key], dtype=torch.long) for item in batch],
                batch_first=True,
                padding_value=value
            )

        return {
            "input_ids": pad("input_ids", self.pad_token_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100)
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

        return {
            "input_text": self.clean_text(input_text),
            "label_text": self.clean_text(label_text)
        }


def get_dataloader(cfg, split="train", batch_size=2, shuffle=True, num_workers=2, persistent_workers=True):
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
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    return dataloader

