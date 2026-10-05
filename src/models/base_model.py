from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics import MeanMetric
from torchmetrics.text import ROUGEScore, SacreBLEUScore
from torch.optim import AdamW
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

        # Initialize the model architecture and tokenizer
        self.model = self._build_model()
        # from_pretrained returns the model in eval mode and Lightning keeps submodule modes,
        # so switch to train mode explicitly or dropout stays disabled during fine-tuning
        self.model.train()
        self.tokenizer = self._build_tokenizer()

        # Corpus-level metrics: updated every batch, computed once per epoch (separate state for val and test)
        self.eval_metrics = nn.ModuleDict({stage: self._build_metric() for stage in ("val", "test")})
        self.eval_losses = nn.ModuleDict({stage: MeanMetric() for stage in ("val", "test")})

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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=input_ids.size(0))
        return loss

    def _build_metric(self):
        if self.cfg.params.task == "summarization":
            # rougeLsum (on by default) needs nltk punkt data; only these three are logged
            return ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"))
        if self.cfg.params.task == "translation":
            return SacreBLEUScore()
        return MeanMetric()  # placeholder so every task has the same structure

    def _shared_eval_step(self, batch, stage):
        """Loss, generation and metric updates shared by validation and test."""
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        batch_size = input_ids.size(0)
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.eval_losses[stage].update(loss, batch_size)

        if self.cfg.params.task not in ("summarization", "translation"):
            return loss

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.params.max_target_length,
            num_beams=self.cfg.params.num_beams
        )
        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        label_ids = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
        refs = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if self.cfg.params.task == "translation":
            self.eval_metrics[stage].update(preds, [[ref] for ref in refs])
        else:
            self.eval_metrics[stage].update(preds, refs)
        return loss

    def _log_epoch_metrics(self, stage):
        # exp of the mean loss over the whole epoch, not the mean of per-batch perplexities
        self.log(f"{stage}_perplexity", torch.exp(self.eval_losses[stage].compute()), prog_bar=True)
        self.eval_losses[stage].reset()

        metric = self.eval_metrics[stage]
        if self.cfg.params.task == "summarization":
            scores = metric.compute()
            for key in ("rouge1", "rouge2", "rougeL"):
                self.log(f"{stage}_{key}_fmeasure", scores[f"{key}_fmeasure"], prog_bar=True)
        elif self.cfg.params.task == "translation":
            self.log(f"{stage}_bleu", metric.compute(), prog_bar=True)
        metric.reset()

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, "val")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, "test")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler from the model's trainer params."""
        params = self.cfg.trainer.params
        optimizer = AdamW(self.parameters(), lr=params.learning_rate, weight_decay=params.get("weight_decay", 0.01))
        scheduler = StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)
        return [optimizer], [scheduler]
