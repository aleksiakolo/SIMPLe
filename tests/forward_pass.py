import pytest
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from transformers import AutoTokenizer, MBart50Tokenizer

@pytest.fixture(scope="module")
def hydra_initialize():
    """Fixture to initialize the Hydra environment."""
    with initialize(config_path="../configs/model"):
        yield

def get_config(model_name):
    """Helper function to load a configuration based on model name."""
    cfg = compose(config_name=model_name)
    return cfg

def generate_dummy_batch(tokenizer, input_text, max_length=128):
    """Generates a dummy batch for testing the forward pass."""
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    inputs["labels"] = inputs["input_ids"].clone()  # for testing purposes using input IDs as labels
    return inputs

def test_forward_pass(model, tokenizer, input_text="This is a test sentence."):
    """Tests the forward pass of a model."""
    model.eval()
    with torch.no_grad():
        dummy_batch = generate_dummy_batch(tokenizer, input_text)
        output = model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
            labels=dummy_batch["labels"]
        )
        assert output is not None, "Forward pass did not return any output"
        print("Forward pass completed successfully with output:", output)

def test_bart_forward_pass(hydra_initialize):
    cfg = get_config("bart")
    model = instantiate(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.name)
    test_forward_pass(model, tokenizer)

def test_legalbert_forward_pass(hydra_initialize):
    cfg = get_config("legal_bert")
    model = instantiate(cfg)
    assert hasattr(model, 'tokenizer'), "Tokenizer not initialized in the model"
    tokenizer = model.tokenizer
    test_forward_pass(model, tokenizer)


def test_t5_summarization_forward_pass(hydra_initialize):
    cfg = get_config("t5_summary")
    model = instantiate(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.name)
    test_forward_pass(model, tokenizer)

def test_mbart_translation_forward_pass(hydra_initialize):
    cfg = get_config("mbart")
    model = instantiate(cfg)
    tokenizer = MBart50Tokenizer.from_pretrained(cfg.params.name) 
    test_forward_pass(model, tokenizer)

def test_t5_translation_forward_pass(hydra_initialize):
    cfg = get_config("t5_translate")
    model = instantiate(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.name)
    test_forward_pass(model, tokenizer)

if __name__ == "__main__":
    pytest.main(["-v", "tests/forward_pass.py"])
