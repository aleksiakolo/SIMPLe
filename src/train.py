import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.summarization import BARTSummarizationModel, LegalBERTSummarizationModel, T5SummarizationModel
from src.models.translation import XLMRTranslationModel, T5TranslationModel
from src.models.main import RephraseTranslateModel

def get_model(cfg: DictConfig):
    """Select and return the appropriate model based on task type and model name."""
    if cfg.task == "summarization":
        if "bart" in cfg.model_name:
            return BARTSummarizationModel(cfg)
        elif "t5" in cfg.model_name:
            return T5SummarizationModel(cfg)
        elif "legal-bert" in cfg.model_name:
            return LegalBERTSummarizationModel(cfg)
        else:
            raise ValueError("Unsupported model for summarization.")
    
    elif cfg.task == "translation":
        if "t5" in cfg.model_name:
            return T5TranslationModel(cfg)
        elif "xlm-roberta" in cfg.model_name:
            return XLMRTranslationModel(cfg)
        else:
            raise ValueError("Unsupported model for translation.")
    
    elif cfg.task == "rephrase_translate":
        return RephraseTranslateModel(cfg)
    
    else:
        raise ValueError("Unsupported task type specified in config.")

@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Initialize the model based on the configuration
    model = get_model(cfg)

    # Configure PyTorch Lightning trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.params.cache_dir,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min"
    )
    
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gpus=cfg.trainer.gpus,
        precision=cfg.trainer.precision,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Train the model
    trainer.fit(model)

    # Test the model if a test set is specified in the config
    if cfg.get("test_data", False):
        trainer.test(model)

if __name__ == "__main__":
    main()
