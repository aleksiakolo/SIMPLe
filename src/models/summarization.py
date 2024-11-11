import torch
from transformers import AutoModelForSeq2SeqLM, AutoModel
from .base_model import LitBaseModel

class BARTSummarizationModel(LitBaseModel):
    def _build_model(self):
        """Initialize the BART model for summarization."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        return model

    def _initialize_criterion(self):
        """Initialize the criterion for the summarization model."""
        # BART uses a sequence-to-sequence task-specific loss.
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for BART summarization."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


class LegalBERTSummarizationModel(LitBaseModel):
    def _build_model(self):
        """Initialize LEGAL-BERT for summarization."""
        model = AutoModel.from_pretrained(self.cfg.model_name)
        return model

    def _initialize_criterion(self):
        """Initialize the criterion for the LEGAL-BERT model."""
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for LEGAL-BERT summarization."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


class T5SummarizationModel(LitBaseModel):
    def _build_model(self):
        """Initialize the T5 model for summarization."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        return model

    def _initialize_criterion(self):
        """Initialize the criterion for the T5 model."""
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for T5 summarization."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
