import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.nn.functional import cross_entropy
from torchmetrics.text.bleu import BLEUScore

class TranslationDataset(Dataset):
    def __init__(self, dataset_path, model_name, max_input_length):
        """
        Dataset to load and preprocess translation data.
        Args:
            dataset_path (str): Path to the JSON file containing the dataset.
            model_name (str): Pretrained model name for tokenization.
            max_input_length (int): Maximum input sequence length.
        """
        with open(dataset_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_text = f'Translate the text from "{item["source_language"]}" to "{item["target_language"]}": {item["source_text"]}'
        target_text = item["target_text"]

        tokenized_source = self.tokenizer(
            source_text, truncation=True, max_length=self.max_input_length, padding="max_length", return_tensors="pt"
        )
        tokenized_target = self.tokenizer(
            target_text, truncation=True, max_length=self.max_input_length, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": tokenized_source["input_ids"].squeeze(0),
            "attention_mask": tokenized_source["attention_mask"].squeeze(0),
            "labels": tokenized_target["input_ids"].squeeze(0),
        }


def get_dataloader(dataset_path, model_name, max_input_length, batch_size, shuffle, num_workers):
    """
    Create DataLoader for the given dataset.
    Args:
        dataset_path (str): Path to the dataset file.
        model_name (str): Pretrained model name for tokenization.
        max_input_length (int): Maximum input sequence length.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.
    """
    dataset = TranslationDataset(dataset_path, model_name, max_input_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


class TranslationModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bleu = BLEUScore()
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        val_loss = outputs.loss

        preds = self.model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=self.model.config.max_length
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        bleu_score = self.bleu(decoded_preds, decoded_labels)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_bleu", bleu_score, prog_bar=True)
        return {"val_loss": val_loss, "val_bleu": bleu_score}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
