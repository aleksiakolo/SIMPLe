import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from src.utils.dataloader import get_dataloaders
from src.utils.utils import instantiate_callbacks, instantiate_loggers
import pytorch_lightning as pl


def initialize(cfg: DictConfig):
    # Instantiate model; the Lightning trainer handles device placement and the
    # model's configure_optimizers sets up the optimizer and scheduler
    model = instantiate(cfg.model)

    # Instantiate loggers and callbacks
    train_logger = instantiate_loggers(cfg.get("logger"))
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate the trainer using the Hydra configuration
    trainer = instantiate(cfg.trainer, logger=train_logger, callbacks=callbacks)

    # Load, split and tokenize the dataset once for all three dataloaders
    dataloaders = get_dataloaders(cfg.data, batch_size=cfg.params.batch_size)
    return model, trainer, dataloaders


def train_model(cfg: DictConfig):
    logger.info("Starting train_model function...")

    # Initialize the components needed for training
    model, trainer, dataloaders = initialize(cfg)

    if cfg.get("train"):
        # ckpt_path resumes training from a checkpoint when set
        trainer.fit(model, dataloaders.train, dataloaders.val, ckpt_path=cfg.get("ckpt_path"))
        logger.info("Training completed.")

    if cfg.get("test"):
        # Evaluate the best checkpoint (by the checkpoint callback's monitored metric) if one was saved
        ckpt_path = "best" if cfg.get("train") and trainer.checkpoint_callback else cfg.get("ckpt_path")
        if ckpt_path == "best" and not trainer.checkpoint_callback.best_model_path:
            ckpt_path = None
        logger.info(f"Testing with checkpoint: {ckpt_path or 'current weights'}")
        trainer.test(model, dataloaders.test, ckpt_path=ckpt_path)

    return model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Log the configuration for debugging
    logger.info(f"Running training with {cfg.model.params.name } on the dataset {cfg.data.dataset.name}.")
    pl.seed_everything(cfg.seed, workers=True)
    train_model(cfg)

if __name__ == "__main__":
    main()
