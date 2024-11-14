import dataclasses
import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.dataloader import get_dataloader
from src.utils.utils import instantiate_loggers


@dataclasses.dataclass
class Dataloaders:
    dataload_eval_train: torch.utils.data.DataLoader
    dataload_eval_val: torch.utils.data.DataLoader
    dataload_eval_test: torch.utils.data.DataLoader


def initialize(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model
    model = instantiate(cfg.model).to(device)
    
    # Load pretrained model from checkpoint if specified
    if cfg.eval.checkpoint_path:
        model = model.load_from_checkpoint(cfg.eval.checkpoint_path, cfg=cfg).to(device)
        logger.info(f"Loaded pretrained model from {cfg.eval.checkpoint_path}")
    else:
        logger.warning("No checkpoint path specified. Using the model with default initialization.")
    
    # Instantiate logger
    eval_logger = instantiate_loggers(cfg.get("logger"))
    
    # Create the PyTorch Lightning Trainer for evaluation
    trainer = instantiate(cfg.trainer, logger=eval_logger, callbacks=[], enable_checkpointing=False)
    
    # Load evaluation dataloaders
    dataload_eval_train = get_dataloader(cfg, split="train", batch_size=cfg.params.batch_size, shuffle=False)
    dataload_eval_val = get_dataloader(cfg, split="val", batch_size=cfg.params.batch_size, shuffle=False)
    dataload_eval_test = get_dataloader(cfg, split="test", batch_size=cfg.params.batch_size, shuffle=False)
    
    dataloaders = Dataloaders(dataload_eval_train, dataload_eval_val, dataload_eval_test)
    return model, trainer, dataloaders, device


def evaluate_model(cfg: DictConfig):
    logger.info("Starting evaluate_model function...")
    
    # Initialize the components needed for evaluation
    model, trainer, dataloaders, device = initialize(cfg)
    
    # Run evaluation on the validation and test sets
    if dataloaders.dataload_eval_val:
        logger.info("Evaluating on validation set...")
        trainer.validate(model, dataloaders.dataload_eval_val)
    
    if dataloaders.dataload_eval_test:
        logger.info("Evaluating on test set...")
        trainer.test(model, dataloaders.dataload_eval_test)
    
    logger.info("Evaluation completed.")


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    # Log the configuration for debugging
    logger.info(f"Running evaluation with configuration: {cfg}")
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
