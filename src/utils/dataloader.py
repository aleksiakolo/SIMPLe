import dataclasses
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
        input_text = examples["translation"].get(source) or ""
        label_text = examples["translation"].get(target) or ""

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
        max_target_length = self.cfg.tokenization.get("max_target_length", max_length)
        logger.info("Tokenizing the dataset ...")

        # T5 needs a task prefix such as "summarize: " or "translate English to Spanish: "
        input_prefix = self.cfg.tokenization.get("input_prefix") or ""

        def tokenize_function(batch):
            # Tokenize input and label as single sequences; padding happens per batch in collate_fn
            inputs = tokenizer(
                [input_prefix + text for text in batch["input_text"]],
                max_length=max_length,
                truncation=True
            )
            labels = tokenizer(
                text_target=batch["label_text"],
                max_length=max_target_length,
                truncation=True
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels["input_ids"]
            }

        return dataset.map(tokenize_function, batched=True)


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
        # Drop examples with a missing source or target (e.g. Multi-LexSum cases without a short/tiny summary)
        before = len(preprocessed)
        preprocessed = preprocessed.filter(lambda ex: bool(ex["input_text"]) and bool(ex["label_text"]))
        if len(preprocessed) < before:
            logger.info(f"Dropped {before - len(preprocessed)} examples with an empty source or target")
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

    def prepare_split(self, split_ds):
        """Each TED example is a whole talk, so truncating it would cut source and target at
        unrelated points. Split talks into line-aligned sentence pairs first; talks whose source
        and target line counts differ cannot be aligned and are dropped."""
        source, target = self.cfg.dataset.source, self.cfg.dataset.target

        def split_talks(batch):
            pairs = []
            for talk in batch["translation"]:
                src_lines = [line for line in (talk.get(source) or "").split("\n") if line.strip()]
                tgt_lines = [line for line in (talk.get(target) or "").split("\n") if line.strip()]
                if len(src_lines) == len(tgt_lines):
                    pairs.extend({source: s, target: t} for s, t in zip(src_lines, tgt_lines))
            return {"translation": pairs}

        talks = len(split_ds)
        split_ds = split_ds.map(split_talks, batched=True, remove_columns=split_ds.column_names)
        if len(split_ds) == 0:
            raise ValueError("No TED talk could be split into line-aligned sentence pairs")
        logger.info(f"Split {talks} talks into {len(split_ds)} aligned sentence pairs")
        return super().prepare_split(split_ds)

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
        # Not every case has every summary length; missing ones are None and get filtered out
        label_text = examples[f"summary/{summary_length}"] or ""

        return {
            "input_text": self.clean_text(input_text),
            "label_text": self.clean_text(label_text)
        }


@dataclasses.dataclass
class Dataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


DATASET_CLASSES = {
    "Helsinki-NLP/europarl": EuroParlDataset,
    "IWSLT/ted_talks_iwslt": TedTalksDataset,
    "allenai/multi_lexsum": MultiLexSumDataset,
}


def get_dataloaders(cfg, batch_size=2, num_workers=2):
    """
    Build train/val/test DataLoaders for the dataset named in the data configuration.
    The dataset is loaded, split, preprocessed and tokenized once and shared by all three loaders.

    Args:
        cfg: Data configuration object.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of workers for data loading.

    Returns:
        Dataloaders: train (shuffled), val and test DataLoaders.
    """
    if cfg.dataset.name not in DATASET_CLASSES:
        raise ValueError(f"Unsupported dataset name: {cfg.dataset.name}")
    dataset = DATASET_CLASSES[cfg.dataset.name](cfg)
    train_ds, val_ds, test_ds = dataset.process_and_save()

    def make_loader(split_ds, shuffle):
        return DataLoader(
            split_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )

    return Dataloaders(make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False))
