import os
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer
import pytorch_lightning as pl


class SummarizationTranslationModel(pl.LightningModule):
    def __init__(self, summarization_model, translation_model_name, device="cuda"):
        """
        Model to handle summarization and translation.
        Args:
            summarization_model: LegalBERT Seq2Seq model for summarization.
            translation_model_name: Name of the pre-trained translation model (e.g., "google/mt5-base").
            device: Device to load the model (default: "cuda").
        """
        super().__init__()
        self.summarization_model = summarization_model
        self.summarization_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.translation_tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large").to(device)
        self.device = device

    def forward(self, sources, input_ids, attention_mask):
        """
        Forward pass for summarization and translation.
        Args:
            sources: Embedded input sources for summarization.
            input_ids: Input IDs for the translation model.
            attention_mask: Attention mask for the translation model.
        Returns:
            Translated outputs.
        """
        # Summarize using LegalBERTSeq2Seq
        summarization_output = self.summarization_model(sources, input_ids, attention_mask)

        # Decode the summarized output into text
        summarized_text = self.summarization_tokenizer.batch_decode(
            summarization_output["logits"].argmax(dim=-1), skip_special_tokens=True
        )

        # Create translation prompts with special word handling
        translation_inputs = self._prepare_translation_prompts(summarized_text)

        # Tokenize translation inputs
        tokenized_translation = self.translation_tokenizer(
            translation_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        # Translate
        translated_output = self.translation_model.generate(
            input_ids=tokenized_translation["input_ids"],
            attention_mask=tokenized_translation["attention_mask"],
            max_length=512,
        )

        # Decode translated text
        translated_text = self.translation_tokenizer.batch_decode(translated_output, skip_special_tokens=True)

        return translated_text

    def _prepare_translation_prompts(self, summarized_texts):
        """
        Prepare translation prompts by embedding special words.
        Args:
            summarized_texts: List of summarized texts.
        Returns:
            List of translation prompts.
        """
        prompts = []
        for text in summarized_texts:
            # Extract special words in the text using a placeholder strategy (e.g., words in quotes)
            special_words = self._extract_special_words(text)
            prompt = (
                f"Translate the following text to Spanish, keeping the special words untranslated: "
                f"Special words: {', '.join(special_words)}\n\n{text}"
            )
            prompts.append(prompt)
        return prompts

    def _extract_special_words(self, text):
        """
        Extract special words from a text that should not be translated. 
        """
        honorifics = {
        "Mr.", "Mrs.", "Miss", "Ms.", "Mx.", "Sir", "Dame", "Dr.", "Prof.", "Lord", "Lady",
        "Sr.", "Sra.", "Srta.", "Don", "Doña", "Profesor", "Profesora", "Dott.", "Dott.ssa",
        "Sig.", "Sig.ra", "Sig.na", "Herr", "Frau", "Fräulein", "Maître", "Freiherr", "Freifrau",
        "Monsieur", "Madame", "Mademoiselle", "M.", "Mme.", "Mlle.", "Eng.", "De Heer", "Mevrouw",
        "Mejuffrouw", "Г-н", "Г-жа", "Г-ца", "Tovarishch", "Pan", "Pani", "Panna", "Herr", "Fru",
        "Fröken", "Frk.", "Tohtori", "Professori", "Kyrios", "Kyria", "Kathigitis", "Kathigitria",
        "Úr", "Asszony", "Kisasszony", "Paní", "Slečna", "Domnul", "Doamna", "Domnișoara",
        "Bey", "Bay", "Hanım", "Baron", "Baroness", "Duke", "Duchess", "Marquis", "Marquise",
        "Frau Doktor", "Herr Doktor", "Herr Professor", "Fru Doktor", "Herrn", "Dr.med.", "Ing."
        }

        # Split the text into tokens
        tokens_punc = text.split(' ')
        tokens = []

        # Split eos punctuations unless it's part of an honorific.
        for token in tokens_punc:
            if token[-1] in {'.', '?', '!'}:
                if token not in honorifics:
                    tokens.append(token[:-1])
                    tokens.append(token[-1])
                else:
                    tokens.append(token)
            else:
                tokens.append(token)

        # To get proper nouns mid-sentence
        pns = []
        for i, token in enumerate(tokens):
            if re.match(r'^[A-Z][a-z\'\-]*$', token): # Capitalized
                if i != 0: # Not start of sequence
                    if tokens[i - 1] not in {'.', '!', '?'}: # Not start of sentence
                        if token not in honorifics: # Not honorific
                            if token not in pns: # Not tagged already
                                pns.append(token)

        return ', '.join(pns)

    def training_step(self, batch, batch_idx):
        """
        Training step for summarization and translation.
        """
        sources = batch["sources"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Summarization
        summarization_outputs = self.summarization_model(sources, input_ids, attention_mask)
        summarized_logits = summarization_outputs["logits"]

        # Decode summarized text
        summarized_text = self.translation_tokenizer.batch_decode(
            summarized_logits.argmax(dim=-1), skip_special_tokens=True
        )

        # Prepare translation prompts and tokenize
        translation_inputs = self._prepare_translation_prompts(summarized_text)
        tokenized_translation = self.translation_tokenizer(
            translation_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        # Translation with labels
        translation_outputs = self.translation_model(
            input_ids=tokenized_translation["input_ids"],
            attention_mask=tokenized_translation["attention_mask"],
            labels=batch["labels"],
        )

        # Calculate translation loss
        translation_loss = translation_outputs.loss
        self.log("translation_loss", translation_loss, prog_bar=True)

        return translation_loss

    def configure_optimizers(self):
        """
        Configure optimizers for the model.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
