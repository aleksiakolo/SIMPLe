import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import get_dataloader
from models.translation_model import TranslationModel


@hydra.main(version_base=None, config_path="configs", config_name="translation")
def train_model(cfg: DictConfig):
    # Load DataLoaders
    train_loader = get_dataloader(cfg.paths.dataset_train, cfg.model.name, cfg.model.max_input_length, cfg.trainer.batch_size, True, cfg.trainer.num_workers)
    val_loader = get_dataloader(cfg.paths.dataset_val, cfg.model.name, cfg.model.max_input_length, cfg.trainer.batch_size, False, cfg.trainer.num_workers)

    # Initialize Model
    model = TranslationModel(cfg.model.name, cfg.trainer.learning_rate)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=3, dirpath=cfg.paths.checkpoints, filename="{epoch}-{val_loss:.2f}"
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=cfg.trainer.patience, mode="min")

    # Logger
    logger_tb = TensorBoardLogger(save_dir=cfg.paths.logs, name="translation_model")

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger_tb,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_model()
