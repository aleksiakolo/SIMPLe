import os

import pytest
from omegaconf import OmegaConf
from src.utils.dataloader import get_dataloaders

pytestmark = pytest.mark.integration

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

CASES = [
    ("europarl", "t5_translate"),
    ("europarl", "mbart"),
    ("ted_talks", "t5_translate"),
    ("ted_talks", "mbart"),
    ("lex_sum", "bart"),
    ("lex_sum", "legal_bert"),
    ("lex_sum", "t5_summary"),
]


@pytest.mark.parametrize("data_name, model_name", CASES)
def test_dataloader_batches(data_name, model_name):
    cfg = OmegaConf.create({
        "data": OmegaConf.load(os.path.join(CONFIG_DIR, "data", f"{data_name}.yaml")),
        "model": OmegaConf.load(os.path.join(CONFIG_DIR, "model", f"{model_name}.yaml")),
    })
    cfg.data.dataset.subset_fraction = 0.001  # keep the test fast
    dataloaders = get_dataloaders(cfg.data, batch_size=4, num_workers=0)

    batch = next(iter(dataloaders.train))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].shape[0] == batch["input_ids"].shape[0]
