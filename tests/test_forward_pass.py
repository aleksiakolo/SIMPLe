import pytest
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def hydra_initialize():
    """Fixture to initialize the Hydra environment."""
    with initialize(version_base=None, config_path="../configs/model"):
        yield


def run_forward_pass(model, input_text="This is a test sentence."):
    """Runs a forward pass with the input as its own label and returns the loss."""
    model.eval()
    with torch.no_grad():
        inputs = model.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"].clone()  # for testing purposes using input IDs as labels
        )
    return output.loss


@pytest.mark.parametrize("config_name", ["bart", "legal_bert", "t5_summary", "mbart", "t5_translate"])
def test_forward_pass(hydra_initialize, config_name):
    model = instantiate(compose(config_name=config_name))
    loss = run_forward_pass(model)
    assert torch.isfinite(loss), f"Forward pass for {config_name} returned a non-finite loss"
