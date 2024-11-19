import dataclasses
import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from src.utils.dataloader import get_dataloader
from src.utils.utils import instantiate_callbacks, instantiate_loggers
import pytorch_lightning as pl


@dataclasses.dataclass
class Dataloaders:
    dataload_train: torch.utils.data.DataLoader
    dataload_eval_train: torch.utils.data.DataLoader
    dataload_eval_val: torch.utils.data.DataLoader
    dataload_eval_test: torch.utils.data.DataLoader


def initialize(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model and move it to the device
    model = instantiate(cfg.model).to(device)
    
    # Instantiate optimizer and scheduler using Hydra utils
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    
    # Instantiate loggers and callbacks
    train_logger = instantiate_loggers(cfg.get("logger"))
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    
    # Instantiate the trainer using the Hydra configuration
    trainer = instantiate(cfg.trainer, logger=train_logger, callbacks=callbacks)
    
    # Load dataloaders for training and evaluation
    dataload_train = get_dataloader(cfg.data, split="train", batch_size=cfg.params.batch_size)
    dataload_eval_train = get_dataloader(cfg.data, split="val", batch_size=cfg.params.batch_size)
    dataload_eval_val = get_dataloader(cfg.data, split="val", batch_size=cfg.params.batch_size)
    dataload_eval_test = get_dataloader(cfg.data, split="test", batch_size=cfg.params.batch_size)
    
    dataloaders = Dataloaders(dataload_train, dataload_eval_train, dataload_eval_val, dataload_eval_test)
    return model, optimizer, scheduler, trainer, dataloaders, device


def train_model(cfg: DictConfig):
    logger.info("Starting train_model function...")
    
    # Initialize the components needed for training
    model, optimizer, scheduler, trainer, dataloaders, device = initialize(cfg)
    
    # Override the `configure_optimizers` method in the model if needed
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
    }
    
    # Train the model using the trainer
    trainer.fit(model, dataloaders.dataload_train, dataloaders.dataload_eval_train)
    
    logger.info("Training completed.")
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Log the configuration for debugging
    logger.info(f"Running training with {cfg.model.params.name } on the dataset {cfg.data.dataset.name}.")
    train_model(cfg)

if __name__ == "__main__":
    main()
