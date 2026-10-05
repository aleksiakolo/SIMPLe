from transformers import AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MBart50Tokenizer
from src.models.base_model import LitBaseModel
from loguru import logger


class TranslationModel(LitBaseModel):
    def __init__(self, cfg):
        logger.info(f"Initializing {self.__class__.__name__}...")
        super().__init__(cfg)
        logger.info(f"{self.__class__.__name__} initialized successfully.")

    def _build_model(self):
        """Initialize the model for translation."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.params.name)
        logger.info(f"Model {self.cfg.params.name} loaded for {self.__class__.__name__}.")
        return model

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for translation."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class mBART(TranslationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer.src_lang = self.cfg.params.src_lang
        self.tokenizer.tgt_lang = self.cfg.params.tgt_lang
        # Without this mBART-50 does not know which language to generate
        tgt_lang_id = self.tokenizer.lang_code_to_id[self.cfg.params.tgt_lang]
        self.model.config.forced_bos_token_id = tgt_lang_id
        self.model.generation_config.forced_bos_token_id = tgt_lang_id

    def _build_model(self):
        """Initialize the mBART model for translation."""
        model = MBartForConditionalGeneration.from_pretrained(self.cfg.params.name)
        logger.info(f"Model {self.cfg.params.name} loaded for mBART translation model.")
        return model

    def _build_tokenizer(self):
        return MBart50Tokenizer.from_pretrained(self.cfg.params.name)


class T5Translation(TranslationModel):
    """T5 translation model (t5-large by default, see configs/model/t5_translate.yaml)."""
