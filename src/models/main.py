import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from base_model import LitBaseModel

class RephraseTranslateModel(LitBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rephrase_model_name = cfg.rephrase_model_name
        self.translate_model_name = cfg.translate_model_name

        # Load the rephrasing model and tokenizer
        self.rephrase_tokenizer = AutoTokenizer.from_pretrained(self.rephrase_model_name)
        self.rephrase_model = AutoModel.from_pretrained(self.rephrase_model_name)

        # Load the translation model and tokenizer
        self.translate_tokenizer = AutoTokenizer.from_pretrained(self.translate_model_name)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(self.translate_model_name)

    def _build_model(self):
        """No specific build needed as we are using pre-initialized models."""
        return None

    def _initialize_criterion(self):
        """Initialize a placeholder criterion for compatibility."""
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        """Run rephrasing and translation sequentially."""
        # Step 1: Rephrase and get embeddings
        with torch.no_grad():
            encoder_outputs = self.rephrase_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = encoder_outputs.last_hidden_state  # Extract embeddings from the rephrase model

        # Step 2: Pass embeddings to the translation model
        translated_ids = self.translate_model.generate(
            inputs_embeds=embeddings,
            max_length=self.cfg.max_output_length,
            num_beams=self.cfg.num_beams
        )

        return translated_ids

    def generate_rephrased_translation(self, text, max_length=512):
        """Generate rephrased embeddings and use them for translation."""
        # Tokenize the input text for the rephrase model
        inputs = self.rephrase_tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)

        attention_mask = (inputs != self.rephrase_tokenizer.pad_token_id).long().to(self.device)

        # Get the translated text
        translated_ids = self.forward(inputs, attention_mask)
        translated_text = self.translate_tokenizer.decode(translated_ids[0], skip_special_tokens=True)

        return translated_text
