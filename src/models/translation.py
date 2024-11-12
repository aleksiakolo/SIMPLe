import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel
from base_model import LitBaseModel

class XLMRTranslationModel(LitBaseModel):
    def _build_model(self):
        """Initialize the XLM-R model for translation."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        return model

    def _initialize_criterion(self):
        """Initialize the criterion for the translation model."""
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for XLM-R translation."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


class T5TranslationModel(LitBaseModel):
    def _build_model(self):
        """Initialize the T5 model for translation."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        return model

    def _initialize_criterion(self):
        """Initialize the criterion for the T5 model."""
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for T5 translation."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
