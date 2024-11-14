from dotenv import load_dotenv
import os
import sys
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from src.utils.dataloader import EuroParlDataset, TedTalksDataset, MultiLexSumDataset
from transformers import AutoTokenizer

# Load environment variables from the .env file
load_dotenv()

# Add project root to sys.path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_dataloader_with_tokenizer(cfg_data, model_cfgs, dataset_cls):
    print(f"Testing DataLoader for {cfg_data.dataset.name}...")
    for model_cfg in model_cfgs:
        cfg_data.tokenization.model_name = model_cfg.params.name
        print(f"Testing with tokenizer: {cfg_data.tokenization.model_name}")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg_data.tokenization.model_name)

        # Create dataset instance
        dataset = dataset_cls(cfg_data)

        # Process and save the dataset
        train_ds, val_ds, test_ds = dataset.process_and_save()
        
        # Print the sizes of the splits
        print(f"{cfg_data.dataset.name} Training set size: {len(train_ds)}")
        print(f"{cfg_data.dataset.name} Validation set size: {len(val_ds)}")
        print(f"{cfg_data.dataset.name} Test set size: {len(test_ds)}")

        dataloader = DataLoader(train_ds, batch_size=16, collate_fn=dataset.collate_fn)
        
        for batch in dataloader:
            # Print keys and assert checks for testing
            print(f"{cfg_data.dataset.name} batch keys:", batch.keys())
            assert "input_ids" in batch, "input_ids missing in batch"
            assert "attention_mask" in batch, "attention_mask missing in batch"
            assert "labels" in batch, "labels missing in batch"

            # Print shapes to ensure data structure is correct
            print(f"{cfg_data.dataset.name} input_ids shape:", batch["input_ids"].shape)
            print(f"{cfg_data.dataset.name} attention_mask shape:", batch["attention_mask"].shape)
            print(f"{cfg_data.dataset.name} labels shape:", batch["labels"].shape)
            break

if __name__ == "__main__":
    # Load dataset configs
    europarl_cfg = OmegaConf.load("configs/data/europarl.yaml")
    ted_talks_cfg = OmegaConf.load("configs/data/ted_talks.yaml")
    multi_lexsum_cfg = OmegaConf.load("configs/data/lex_sum.yaml")

    # Load model configs
    t5_cfg = OmegaConf.load("configs/model/t5_translate.yaml")
    mbart_cfg = OmegaConf.load("configs/model/mbart.yaml")
    bart_cfg = OmegaConf.load("configs/model/bart.yaml")
    legalbert_cfg = OmegaConf.load("configs/model/legal_bert.yaml")
    t5_sum_cfg = OmegaConf.load("configs/model/t5_summary.yaml")

    # Define the model configurations for each dataset
    europarl_model_cfgs = [t5_cfg, mbart_cfg]
    ted_talks_model_cfgs = [t5_cfg, mbart_cfg]
    multi_lexsum_model_cfgs = [bart_cfg, legalbert_cfg, t5_sum_cfg]

    test_dataloader_with_tokenizer(europarl_cfg, europarl_model_cfgs, EuroParlDataset)

    test_dataloader_with_tokenizer(ted_talks_cfg, ted_talks_model_cfgs, TedTalksDataset)

    test_dataloader_with_tokenizer(multi_lexsum_cfg, multi_lexsum_model_cfgs, MultiLexSumDataset)
