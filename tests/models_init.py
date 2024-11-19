import pytest
from omegaconf import OmegaConf
from hydra import initialize, compose
from src.models.summarization import BART, LegalBERT, T5Summarization
from src.models.translation import mBART, T5Translation  
from hydra.utils import instantiate

@pytest.fixture(scope="module")
def hydra_initialize():
    """Fixture to initialize the Hydra environment."""
    with initialize(config_path="../configs/model"):
        yield

def get_config(model_name):
    """Helper function to load a configuration based on model name."""
    cfg = compose(config_name=model_name)
    return cfg

def test_bart_initialization(hydra_initialize):
    """Test initialization of the BART model."""
    try:
        cfg = get_config("bart")
        model = instantiate(cfg)  
        assert isinstance(model, BART), "BART model is not instantiated as the correct class."
        print("BART model initialized successfully.")
    except Exception as e:
        pytest.fail(f"BART model initialization failed: {e}")

def test_legalbert_initialization(hydra_initialize):
    """Test initialization of the LegalBERT model."""
    try:
        cfg = get_config("legal_bert")
        model = instantiate(cfg)  
        assert isinstance(model, LegalBERT), "LegalBERT model is not instantiated as the correct class."
        print("LegalBERT model initialized successfully.")
    except Exception as e:
        pytest.fail(f"LegalBERT model initialization failed: {e}")

def test_t5_summarization_initialization(hydra_initialize):
    """Test initialization of the T5 model for summarization."""
    try:
        cfg = get_config("t5_summary")
        model = instantiate(cfg)  
        assert isinstance(model, T5Summarization), "T5 model is not instantiated as the correct class."
        print("T5 model initialized successfully.")
    except Exception as e:
        pytest.fail(f"T5 model initialization failed: {e}")

def test_mbart_translation_initialization(hydra_initialize):
    """Test initialization of the mBART model for translation."""
    try:
        cfg = get_config("mbart")
        model = instantiate(cfg)  
        assert isinstance(model, mBART), "mBART model is not instantiated as the correct class."
        print("mBART model initialized successfully.")
    except Exception as e:
        pytest.fail(f"mBART model initialization failed: {e}")

def test_t5_translation_initialization(hydra_initialize):
    """Test initialization of the T5 model for translation."""
    try:
        cfg = get_config("t5_translate")
        model = instantiate(cfg)  
        assert isinstance(model, T5Translation), "T5 model is not instantiated as the correct class."
        print("T5 model for translation initialized successfully.")
    except Exception as e:
        pytest.fail(f"T5 model for translation initialization failed: {e}")

if __name__ == "__main__":
    pytest.main(["-v", "tests/models_init.py"])