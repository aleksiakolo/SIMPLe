import torch
from transformers import AutoModelForSeq2SeqLM, EncoderDecoderModel, GPT2Config, GPT2LMHeadModel, BertModel
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
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id or self.tokenizer.eos_token_id or 0
        self.model.config.pad_token_id = self.tokenizer.pad_token_id or 0

    def _build_model(self):
        """Initialize LEGAL-BERT as the encoder and a compatible decoder."""
        logger.info(f"Building encoder-decoder model using {self.cfg.params.name} as the encoder...")

        # Load Legal-BERT as the encoder
        encoder = BertModel.from_pretrained(self.cfg.params.name)

        # Compatible decoder configuration
        decoder_config = GPT2Config.from_pretrained("gpt2")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True # Cross-attention for seq2seq

        # Initialize the decoder
        decoder = GPT2LMHeadModel(config=decoder_config)

        # Create an EncoderDecoderModel
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        logger.info(f"Encoder-decoder model with {self.cfg.params.name} as the encoder built successfully.")
        return model


class T5Summarization(SummarizationModel):
    """T5 summarization model (t5-large by default, see configs/model/t5_summary.yaml)."""
