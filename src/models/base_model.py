from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

import torch
import torch.nn as nn
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer

T = TypeVar("T", bound="Module")


class Module(ABC):
    @classmethod
    def initialize(cls: type[T], **kwargs) -> T:
        return cls(DictConfig(kwargs, flags={"allow_objects": True}))


class LitBaseModel(Module, pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        self.logging_frequency = getattr(self.cfg.params, "logging_frequency", 10)

        # Initialize the model architecture and tokenizer
        self.model = self._build_model()
        self.tokenizer = self._build_tokenizer()

        # Initialize metrics based on task
        if self.cfg.params.task == "summarization":
            # rougeLsum (on by default) needs nltk punkt data; only these three are logged
            self.rouge = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"))
        elif self.cfg.params.task == "translation":
            self.bleu = BLEUScore()

        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Abstract method for constructing the model architecture."""
        pass

    def _build_tokenizer(self):
        """Load the tokenizer matching the model. The loss is computed by the HF model itself."""
        return AutoTokenizer.from_pretrained(self.cfg.params.name)

    def training_step(self, batch, batch_idx):
        """Defines the training step."""
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def calculate_perplexity(self, loss: torch.Tensor) -> torch.Tensor:
        """Calculates perplexity from the mean per-token cross-entropy loss."""
        return torch.exp(loss)
    
    def log_val_outputs(self, outputs, stage="Validation"):
        """Logs validation step outputs."""
        logger.info(f"{stage} Outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            logger.info(f"{stage} {key}: {value}")
        self.log_dict(outputs, on_epoch=True, prog_bar=True)

    def log_test_outputs(self, outputs, stage="Test"):
        """Logs test step outputs."""
        logger.info(f"{stage} Outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            logger.info(f"{stage} {key}: {value}")
        self.log_dict(outputs, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Defines the validation step with task-specific metrics."""
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss

        # HF loss is already the mean per-token cross-entropy, so perplexity is just exp(loss)
        perplexity = torch.exp(val_loss)
        self.log('val_perplexity', perplexity, on_epoch=True, prog_bar=True)

        # Generate outputs for metric calculations
        generated_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.cfg.params.max_input_length
        )

        # Decode outputs and labels for comparison
        decoded_preds = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        label_ids = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Log metrics based on the task
        if self.cfg.params.task == "summarization":
            rouge_scores = self.rouge(decoded_preds, decoded_labels)
            outputs_dict = {
                'val_loss': val_loss,
                'val_perplexity': perplexity,
                'val_rouge1_fmeasure': rouge_scores['rouge1_fmeasure'],
                'val_rouge2_fmeasure': rouge_scores['rouge2_fmeasure'],
                'val_rougeL_fmeasure': rouge_scores['rougeL_fmeasure']
            }
        elif self.cfg.params.task == "translation":
            avg_bleu_score = self.bleu(decoded_preds, decoded_labels)
            outputs_dict = {
                'val_loss': val_loss,
                'val_perplexity': perplexity,
                'val_bleu': avg_bleu_score
            }
        else:
            outputs_dict = {'val_loss': val_loss}

        self.log_val_outputs(outputs_dict)
        return outputs_dict

    def test_step(self, batch, batch_idx):
        """Defines the test step with task-specific metrics."""
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        test_loss = outputs.loss

        # HF loss is already the mean per-token cross-entropy, so perplexity is just exp(loss)
        perplexity = torch.exp(test_loss)
        self.log('test_perplexity', perplexity, on_epoch=True, prog_bar=True)

        # Generate outputs for metric calculations
        generated_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.cfg.params.max_input_length
        )

        # Decode outputs and labels for comparison
        decoded_preds = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        label_ids = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Log metrics based on the task
        if self.cfg.params.task == "summarization":
            rouge_scores = self.rouge(decoded_preds, decoded_labels)
            outputs_dict = {
                'test_loss': test_loss,
                'test_perplexity': perplexity,
                'test_rouge1_fmeasure': rouge_scores['rouge1_fmeasure'],
                'test_rouge2_fmeasure': rouge_scores['rouge2_fmeasure'],
                'test_rougeL_fmeasure': rouge_scores['rougeL_fmeasure']
            }
        elif self.cfg.params.task == "translation":
            avg_bleu_score = self.bleu(decoded_preds, decoded_labels)
            outputs_dict = {
                'test_loss': test_loss,
                'test_perplexity': perplexity,
                'test_bleu': avg_bleu_score
            }
        else:
            outputs_dict = {'test_loss': test_loss}

        self.log_test_outputs(outputs_dict)
        return outputs_dict

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        optimizer = Adam(self.parameters(), lr=self.cfg.trainer.lr)
        scheduler = StepLR(
            optimizer, step_size=self.cfg.trainer.lr_step_size, gamma=self.cfg.trainer.lr_gamma
        )
        return [optimizer], [scheduler]
