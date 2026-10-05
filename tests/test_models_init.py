import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
from src.models.summarization import BART, LegalBERT, T5Summarization
from src.models.translation import mBART, T5Translation

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def hydra_initialize():
    """Fixture to initialize the Hydra environment."""
    with initialize(version_base=None, config_path="../configs/model"):
        yield


@pytest.mark.parametrize("config_name, model_cls", [
    ("bart", BART),
    ("legal_bert", LegalBERT),
    ("t5_summary", T5Summarization),
    ("mbart", mBART),
    ("t5_translate", T5Translation),
])
def test_model_initialization(hydra_initialize, config_name, model_cls):
    """Each model config instantiates the expected class with a tokenizer."""
    model = instantiate(compose(config_name=config_name))
    assert isinstance(model, model_cls)
    assert model.tokenizer is not None
