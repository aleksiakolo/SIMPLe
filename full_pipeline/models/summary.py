import os
import h5py
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

import torch.nn as nn
import torch.nn.functional as F


class LegalBERTEncoder(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=8, ff_dim=2048, num_layers=4, dropout=0.1):
        """
        LegalBERT-based cross-attention encoder.
        Args:
            embedding_dim (int): Dimensionality of embeddings.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimensionality of feed-forward layers.
            num_layers (int): Number of encoder layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, sources, attention_mask):
        """
        Args:
            sources (torch.Tensor): Batched source embeddings, shape (batch_size, max_sources, max_chunks, embedding_dim).
            attention_mask (torch.Tensor): Mask for valid positions, shape (batch_size, max_sources, max_chunks).
        Returns:
            torch.Tensor: Encoded representation of sources, shape (batch_size, max_sources * max_chunks, embedding_dim).
        """
        batch_size, max_sources, max_chunks, embedding_dim = sources.size()

        # Reshape sources and attention_mask to merge sources and chunks into a single sequence
        reshaped_sources = sources.view(batch_size, max_sources * max_chunks, embedding_dim)
        reshaped_attention_mask = attention_mask.view(batch_size, max_sources * max_chunks)

        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            reshaped_sources = layer(
                reshaped_sources.transpose(0, 1),  # (seq_len, batch_size, embedding_dim)
                src_key_padding_mask=~reshaped_attention_mask.bool()  # Shape: (batch_size, seq_len)
            ).transpose(0, 1)  # Back to (batch_size, seq_len, embedding_dim)

        return reshaped_sources


class LegalBERTSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder_model_name="facebook/bart-large"):
        """
        LegalBERT Seq2Seq model with a custom encoder and BART's decoder.
        Args:
            encoder (LegalBERTEncoder): The custom encoder for cross-source attention.
            decoder_model_name (str): Pretrained BART model name for the decoder.
        """
        super().__init__()
        self.encoder = encoder
        
        # Load BART configuration and initialize model
        self.bart_config = BartConfig.from_pretrained(decoder_model_name)
        self.bart = BartForConditionalGeneration.from_pretrained(decoder_model_name)
        
        # Use only the BART decoder
        self.decoder = self.bart.model.decoder

        # Projection layer to map encoder outputs to the decoder's expected hidden size
        self.projection = nn.Linear(encoder.embedding_dim, self.bart_config.d_model)

        # Language modeling head for final token predictions
        self.lm_head = nn.Linear(self.bart_config.d_model, self.bart_config.vocab_size, bias=False)

    def forward(self, sources, source_attention_mask, input_ids, attention_mask):
        """
        Forward pass with a custom encoder and BART decoder.
        Args:
            sources (torch.Tensor): Batched source embeddings, shape (batch_size, max_sources, max_chunks, embedding_dim).
            source_attention_mask (torch.Tensor): Attention mask for the encoder, shape (batch_size, max_sources, max_chunks).
            input_ids (torch.Tensor): Input IDs for the decoder.
            attention_mask (torch.Tensor): Attention mask for the decoder.
        Returns:
            torch.Tensor: Logits for next-token prediction.
        """
        # Pass through the custom encoder
        encoder_hidden_states = self.encoder(sources, source_attention_mask)
        
        # Project encoder hidden states to match the decoder's expected input size
        encoder_hidden_states = self.projection(encoder_hidden_states)

        # Pass through the decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=source_attention_mask.view(sources.size(0), -1),  # Flattened for compatibility
            use_cache=False,  # Set to False for training
        )

        # Compute logits using the language modeling head
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits

    def generate(self, sources, source_attention_mask, max_length=512, num_beams=2):
        """
        Generate summaries using the model.
        Args:
            sources (torch.Tensor): Batched source embeddings, shape (batch_size, max_sources, max_chunks, embedding_dim).
            source_attention_mask (torch.Tensor): Attention mask for the encoder, shape (batch_size, max_sources, max_chunks).
            max_length (int): Maximum length of the generated summaries.
            num_beams (int): Number of beams for beam search.
        Returns:
            list: Generated summaries as token IDs.
        """
        # Pass through the custom encoder
        encoder_hidden_states = self.encoder(sources, source_attention_mask)

        # Project encoder hidden states to match the decoder's expected input size
        encoder_hidden_states = self.projection(encoder_hidden_states)

        # Prepare encoder outputs for BART's internal generate function
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )

        # Generate summaries using BART's generate method
        generated_ids = self.bart.generate(
            input_ids=None,  # No initial decoder input for generation
            encoder_outputs=encoder_outputs,  # Packaged encoder hidden states
            attention_mask=source_attention_mask.view(sources.size(0), -1),  # Flattened attention mask
            max_length=max_length,
            num_beams=num_beams,
        )

        return generated_ids



class SummarizationDataset(Dataset):
    def __init__(self, data_dir, labels_dir, max_output_length=512):
        """
        Dataset for LegalBERT Seq2Seq summarization task.
        Args:
            data_dir (str): Directory containing HDF5 files with source embeddings.
            labels_dir (str): Directory containing JSON files with tokenized summaries.
            max_output_length (int): Maximum output length for the decoder.
        """
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.max_output_length = max_output_length

        # Collect valid file names from the labels directory
        self.files = [
            f for f in os.listdir(labels_dir) if f.endswith(".json")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        h5_path = os.path.join(self.data_dir, file_name.replace(".json", ".h5"))
        label_path = os.path.join(self.labels_dir, file_name)

        # Load HDF5 source embeddings
        with h5py.File(h5_path, "r") as h5_file:
            sources = [torch.tensor(h5_file[key][:], dtype=torch.float32) for key in h5_file.keys() if key.startswith("source_")]

        # Concatenate sources and create attention mask
        max_chunks = max(source.shape[0] for source in sources)
        padded_sources = torch.zeros((len(sources), max_chunks, sources[0].shape[1]))
        attention_mask = torch.zeros((len(sources), max_chunks), dtype=torch.bool)

        for i, source in enumerate(sources):
            padded_sources[i, :source.shape[0], :] = source
            attention_mask[i, :source.shape[0]] = True

        # Load JSON label
        with open(label_path, "r") as f:
            label_data = json.load(f)

        # Process the label: concatenate chunks into input IDs and attention mask
        chunks = label_data["long"]["chunks"]
        input_ids = []
        label_attention_mask = []
        for chunk_key in sorted(chunks.keys()):
            input_ids.extend(chunks[chunk_key]["input_ids"])
            label_attention_mask.extend(chunks[chunk_key]["attention_mask"])

        # Truncate to max_output_length
        input_ids = input_ids[:self.max_output_length]
        label_attention_mask = label_attention_mask[:self.max_output_length]

        return {
            "sources": padded_sources,
            "source_attention_mask": attention_mask,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(label_attention_mask, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sources and chunks.
    """
    # Determine the maximum number of sources and chunks across the batch
    max_sources = max(len(item["sources"]) for item in batch)
    max_chunks = max(max(source.shape[0] for source in item["sources"]) for item in batch)
    embedding_dim = batch[0]["sources"][0].shape[1]

    batch_size = len(batch)

    # Initialize padded tensors
    padded_sources = torch.zeros((batch_size, max_sources, max_chunks, embedding_dim))
    source_attention_mask = torch.zeros((batch_size, max_sources, max_chunks), dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, source in enumerate(item["sources"]):
            padded_sources[i, j, :source.shape[0], :] = source
            source_attention_mask[i, j, :source.shape[0]] = 1

    # Pad input_ids and attention_mask for the labels
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    return {
        "sources": padded_sources,  # Shape: (batch_size, max_sources, max_chunks, embedding_dim)
        "source_attention_mask": source_attention_mask,  # Shape: (batch_size, max_sources, max_chunks)
        "input_ids": input_ids,  # Shape: (batch_size, max_input_length)
        "attention_mask": attention_mask,  # Shape: (batch_size, max_input_length)
    }



def get_dataloader(data_dir, labels_dir, batch_size, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the dataset.
    Args:
        data_dir (str): Directory containing HDF5 files.
        labels_dir (str): Directory containing JSON labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = SummarizationDataset(data_dir, labels_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
