from transformers import AutoModelForSeq2SeqLM, EncoderDecoderModel
from src.models.base_model import LitBaseModel
from loguru import logger


class SummarizationModel(LitBaseModel):
    def __init__(self, cfg):
        logger.info(f"Initializing {self.__class__.__name__}...")
        super().__init__(cfg)
        logger.info(f"{self.__class__.__name__} initialized successfully.")

    def _build_model(self):
        """Initialize the model for summarization."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.params.name)
        logger.info(f"Model {self.cfg.params.name} loaded for {self.__class__.__name__}.")
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Forward pass for summarization."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class BART(SummarizationModel):
    """BART summarization model (facebook/bart-large-cnn by default, see configs/model/bart.yaml)."""


class LegalBERT(SummarizationModel):
    """Legal-BERT warm-started as both encoder and decoder (BERT2BERT).

    Using Legal-BERT on both sides keeps a single vocabulary and tokenizer, and gives the decoder
    pretrained weights; only the decoder's cross-attention layers start from random initialization.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        config = self.model.config
        config.decoder_start_token_id = self.tokenizer.cls_token_id
        config.eos_token_id = self.tokenizer.sep_token_id
        config.pad_token_id = self.tokenizer.pad_token_id
        config.vocab_size = config.encoder.vocab_size
        generation_config = self.model.generation_config
        generation_config.decoder_start_token_id = self.tokenizer.cls_token_id
        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.sep_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _build_model(self):
        """Build an encoder-decoder model with Legal-BERT as both encoder and decoder."""
        logger.info(f"Building BERT2BERT encoder-decoder model from {self.cfg.params.name}...")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.cfg.params.name, self.cfg.params.name)
        logger.info(f"Encoder-decoder model from {self.cfg.params.name} built successfully.")
        return model


class T5Summarization(SummarizationModel):
    """T5 summarization model (t5-large by default, see configs/model/t5_summary.yaml)."""
