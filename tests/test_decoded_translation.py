import pytest
import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

pytestmark = pytest.mark.integration


def translate(model, tokenizer, sample_text, max_length=128, forced_bos_token_id=None):
    """Generate and decode a translation for a single input text."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sample_text, return_tensors="pt", max_length=max_length, truncation=True)
        outputs = model.generate(**inputs, max_length=max_length, forced_bos_token_id=forced_bos_token_id)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def test_mbart_decodes_non_empty_translation():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="es_XX")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    decoded = translate(model, tokenizer, "This is a test sentence for translation.",
                        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"])
    assert decoded and all(decoded), "Decoded outputs are empty"
