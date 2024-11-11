import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
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

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Initialize the model based on the configuration
    model = get_model(cfg)

    # Load the trained model checkpoint if specified
    if cfg.get("checkpoint_path"):
        model = model.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)

    # Configure PyTorch Lightning trainer for evaluation
    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        precision=cfg.trainer.precision
    )

    # Evaluate the model on the test dataset
    test_results = trainer.test(model)

    # Print test results
    print("Evaluation Results:")
    for key, value in test_results[0].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
