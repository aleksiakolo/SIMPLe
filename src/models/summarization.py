import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderModel, AutoTokenizer, GPT2Config, GPT2LMHeadModel, BertModel
from src.models.base_model import LitBaseModel
from loguru import logger

# class ChunkedSummarizationModel(LitBaseModel):
#     def __init__(self, cfg):
#         logger.info(f"Initializing {self.__class__.__name__}...")
#         super().__init__(cfg)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.params.name)
#         logger.info(f"{self.__class__.__name__} initialized successfully.")

#     def _build_model(self):
#         """Initialize the model for summarization."""
#         model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.params.name)
#         logger.info(f"Model {self.cfg.params.name} loaded for {self.__class__.__name__}.")
#         return model

#     def _initialize_criterion(self):
#         """Initialize the criterion for the model."""
#         logger.info(f"Initializing loss criterion for {self.__class__.__name__}...")
#         return torch.nn.CrossEntropyLoss()

#     def chunk_input(self, input_text, max_length):
#         """Split input text into chunks based on the model's maximum token limit."""
#         logger.info(f"Chunking input text for {self.__class__.__name__}...")
        
#         # Tokenize input text into chunks based on max_length
#         tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
#         print(f"Tokens: {tokens[:20]}...")  
        
#         # Split into chunks
#         chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
#         print(f"Number of chunks: {len(chunks)}")
        
#         # Convert token chunks back to text strings
#         chunked_texts = [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
#         print(f"Decoded chunks: {chunked_texts[:2]}...") 
        
#         logger.info(f"Input text chunked into {len(chunks)} chunks.")
#         return chunked_texts


#     def forward(self, input_ids=None, attention_mask=None, labels=None):
#         """Forward pass with input chunking for long sequences."""
#         max_length = self.cfg.params.max_input_length
#         print(f"Max length: {max_length}")  
#         print(f"Input IDs: {input_ids[:20]}...")  
        
#         # Decode input_ids back to text if they are tensors
#         input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         print(f"Decoded input text: {input_text[:2]}...")  
        
#         # Chunk the input text
#         input_chunks = self.chunk_input(input_text, max_length)
#         print(f"Number of chunks: {len(input_chunks)}")  

#         logger.info(f"Processing {len(input_chunks)} chunks through the model...")
#         outputs = []

#         for idx, chunk in enumerate(input_chunks):
#             print(f"Processing chunk {idx + 1}: {chunk[:50]}...") 

#             # Tokenize each chunk and convert to tensor
#             chunk_input_ids = self.tokenizer(chunk, return_tensors="pt").input_ids.to(self.device)
#             print(f"Chunk input IDs: {chunk_input_ids[:20]}...")  

#             chunk_attention_mask = torch.ones_like(chunk_input_ids) if attention_mask is None else attention_mask
            
#             # Run the model forward pass using the chunked input
#             output = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, labels=labels)
#             outputs.append(output.logits)  # Collect logits
#             print(f"Output logits shape: {output.logits.shape}")  

#         # Concatenate outputs from all chunks
#         final_output = torch.cat(outputs, dim=-1)
#         print(f"Final output shape: {final_output.shape}") 
#         return final_output

class SummarizationModel(LitBaseModel):
    def __init__(self, cfg):
        logger.info(f"Initializing {self.__class__.__name__}...")
        super().__init__(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.params.name)
        self.model = self._build_model()
        logger.info(f"{self.__class__.__name__} initialized successfully.")

    def _build_model(self):
        """Initialize the model for summarization."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.params.name)
        logger.info(f"Model {self.cfg.params.name} loaded for {self.__class__.__name__}.")
        return model

    def _initialize_criterion(self):
        """Initialize the criterion for the model."""
        logger.info(f"Initializing loss criterion for {self.__class__.__name__}...")
        return torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Forward pass for summarization."""
        logger.info("Running forward pass for summarization...")
        if input_ids is not None:
            logger.debug(f"Input IDs shape: {input_ids.shape}")   
        # Forward pass through the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logger.debug(f"Output logits shape: {outputs.logits.shape}")
        return outputs.logits  # Return logits for loss calculation and further processing

class BART(SummarizationModel):
    def __init__(self, cfg):
        logger.info("Initializing BART summarization model...")
        super().__init__(cfg)
        self.cfg.params.name = "facebook/bart-large-cnn"  
        logger.info("BART initialized successfully.")

class LegalBERT(SummarizationModel):
    def __init__(self, cfg):
        logger.info("Initializing LegalBERT seq2seq summarization model...")
        super().__init__(cfg)
        self.cfg.params.name = "nlpaueb/legal-bert-base-uncased"  # Name of the encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.params.name)
        self.model = self._build_model()
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id or self.tokenizer.eos_token_id or 0
        self.model.config.pad_token_id = self.tokenizer.pad_token_id or 0
        logger.info("LegalBERT seq2seq initialized successfully.")

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

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Forward pass for the seq2seq model."""
        logger.info("Running forward pass for LegalBERT seq2seq model...")
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

class T5Summarization(SummarizationModel):
    def __init__(self, cfg):
        logger.info("Initializing T5SummarizationModel...")
        super().__init__(cfg)
        self.cfg.params.name = "t5-large"  
        logger.info("T5SummarizationModel initialized successfully.")
