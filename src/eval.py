import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from src.utils.dataloader import get_dataloaders
from src.utils.utils import instantiate_loggers


def initialize(cfg: DictConfig):
    # The checkpoint weights are loaded by the trainer via ckpt_path
    model = instantiate(cfg.model)

    # Instantiate logger
    eval_logger = instantiate_loggers(cfg.get("logger"))

    # Create the PyTorch Lightning Trainer for evaluation
    trainer = instantiate(cfg.trainer, logger=eval_logger, callbacks=[], enable_checkpointing=False)

    # Load evaluation dataloaders
    dataloaders = get_dataloaders(cfg.data, batch_size=cfg.params.batch_size)
    return model, trainer, dataloaders


def evaluate_model(cfg: DictConfig):
    logger.info("Starting evaluate_model function...")

    # Initialize the components needed for evaluation
    model, trainer, dataloaders = initialize(cfg)

    logger.info(f"Evaluating checkpoint {cfg.ckpt_path} on the validation set...")
    trainer.validate(model, dataloaders.val, ckpt_path=cfg.ckpt_path)

    logger.info(f"Evaluating checkpoint {cfg.ckpt_path} on the test set...")
    trainer.test(model, dataloaders.test, ckpt_path=cfg.ckpt_path)

    logger.info("Evaluation completed.")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    logger.info(f"Running evaluation of {cfg.model.params.name} on the dataset {cfg.data.dataset.name}.")
    pl.seed_everything(cfg.seed, workers=True)
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
