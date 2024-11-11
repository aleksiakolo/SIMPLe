from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

T = TypeVar("T", bound="Module")


class Module(ABC):
    @classmethod
    def initialize(cls: type[T], **kwargs) -> T:
        return cls(DictConfig(kwargs, flags={"allow_objects": True}))


class BaseModel(nn.Module, Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Abstract method for forward pass"""
        pass

    def get_reps(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the encoded representations from the encoder."""
        return self.encoder(x)


class LitBaseModel(Module, pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        self.logging_frequency = getattr(self.cfg.params, "logging_frequency", 10)

        # Initialize the model architecture
        self.model = self._build_model()
        self.criterion = self._initialize_criterion()

        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _build_model(self) -> BaseModel:
        """Abstract method for constructing the model architecture."""
        pass

    @abstractmethod
    def _initialize_criterion(self) -> nn.Module:
        """Abstract method for initializing the loss function."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Lightning module."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Defines the training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation step."""
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Defines the test step."""
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        optimizer = Adam(self.parameters(), lr=self.cfg.trainer.lr)
        scheduler = StepLR(
            optimizer, step_size=self.cfg.trainer.lr_step_size, gamma=self.cfg.trainer.lr_gamma
        )
        return [optimizer], [scheduler]
