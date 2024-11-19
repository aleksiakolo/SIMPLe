import torch
from transformers import AutoModelForSeq2SeqLM, MBartForConditionalGeneration, AutoTokenizer, MBart50Tokenizer
from src.models.base_model import LitBaseModel
from loguru import logger

class mBART(LitBaseModel):
    def __init__(self, cfg):
        logger.info("Initializing mBART translation model...")
        super().__init__(cfg)
        self.tokenizer = MBart50Tokenizer.from_pretrained(self.cfg.params.name)
        logger.info("mBART translation model initialized successfully.")

    def _build_model(self):
        """Initialize the mBART model for translation."""
        model = MBartForConditionalGeneration.from_pretrained(self.cfg.params.name)
        logger.info(f"Model {self.cfg.params.name} loaded for mBART translation model.")
        return model
    
    def _initialize_criterion(self):
        """Initialize the criterion for the translation model."""
        logger.info("Initializing criterion for mBART translation model...")
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for mBART translation."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

class T5Translation(LitBaseModel):
    def __init__(self, cfg):
        logger.info("Initializing T5 translation model...")
        super().__init__(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.params.name)
        logger.info("T5 translation model initialized successfully.")

    def _build_model(self):
        """Initialize the T5 model for translation."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.params.name)
        logger.info(f"Model {self.cfg.params.name} loaded for T5 translation model.")
        return model
    
    def _initialize_criterion(self):
        """Initialize the criterion for the translation model."""
        logger.info("Initializing criterion for T5 translation model...")
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for T5 translation."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


# Needs further work because so far it has mainly been used for classification
# and not generation tasks. The encoder and decoder come separately so the architecture
# needs further work.


# class XLMRTranslationModel(LitBaseModel):
#     def __init__(self, cfg):
#         logger.info("Initializing XLM-R translation model...")
#         super().__init__(cfg)
#         logger.info("XLM-R translation model initialized successfully.")

#     def _build_model(self):
#         """Initialize the XLM-R model for translation."""
#         model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model.name)
#         logger.info(f"Model {self.cfg.model.name} loaded for XLMRTranslationModel.")
#         return model
    
#     def _initialize_criterion(self):
#         """Initialize the criterion for the translation model."""
#         logger.info("Initializing criterion for XLM-R translation model...")
#         return torch.nn.CrossEntropyLoss()

#     def forward(self, input_ids, attention_mask=None, labels=None):
#         """Forward pass for XLM-R translation."""
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
#         return outputs