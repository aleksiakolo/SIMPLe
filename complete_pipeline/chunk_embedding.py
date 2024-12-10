import os
import h5py
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from datasets import load_from_disk

class LegalBERTEmbedder:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", device="cuda"):
        """
        Initialize the LegalBERT model and tokenizer for embeddings.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_chunks(self, tokenized_chunks):
        """
        Embed tokenized chunks using LegalBERT.
        Args:
            tokenized_chunks (list): List of dictionaries with input_ids and attention_mask.
        Returns:
            torch.Tensor: Embedded chunks of shape (num_chunks, embedding_dim).
        """
        embeddings = []
        with torch.no_grad():
            for chunk in tokenized_chunks:
                input_ids = torch.tensor(chunk["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(chunk["attention_mask"]).unsqueeze(0).to(self.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                chunk_embedding = output.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(chunk_embedding.squeeze(0).cpu())
        return torch.stack(embeddings)

def process_example(example, embedder, tokenized_dir, output_dir, split, idx):
    """
    Process a single example to generate embeddings and save them along with labels.
    """
    # Load tokenized JSON file
    tokenized_path = os.path.join(tokenized_dir, split, f"{idx}.json")
    with open(tokenized_path, "r") as f:
        tokenized_data = json.load(f)

    # Create HDF5 file for embeddings
    h5_path = os.path.join(output_dir, "data", split, f"{idx}.h5")
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as h5_file:
        for source_key, source_data in tokenized_data.items():
            if source_key.startswith("source_"):
                chunks = source_data.get("chunks", [])
                embeddings = embedder.embed_chunks(chunks)
                h5_file.create_dataset(source_key, data=embeddings.numpy())  # Save embeddings as a dataset

    # Save labels (target summaries and tokens) as JSON
    labels = {}
    for summary_type in ["long", "small", "tiny"]:
        if summary_type in tokenized_data:
            labels[summary_type] = tokenized_data[summary_type]
            labels[f"{summary_type}_input_ids"] = tokenized_data.get(f"{summary_type}_input_ids", [])
            labels[f"{summary_type}_attention_mask"] = tokenized_data.get(f"{summary_type}_attention_mask", [])

    labels_path = os.path.join(output_dir, "labels", split, f"{idx}.json")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)

def embed_and_save(tokenized_dir, output_dir, model_name="nlpaueb/legal-bert-base-uncased", split="train", sample_size=None):
    """
    Embed tokenized sources and save embeddings (HDF5) and labels (JSON).
    """
    embedder = LegalBERTEmbedder(model_name=model_name)
    tokenized_split_dir = os.path.join(tokenized_dir, split)

    # Get list of tokenized files
    file_list = [f for f in os.listdir(tokenized_split_dir) if f.endswith(".json")]
    if sample_size:
        file_list = file_list[:sample_size]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_example, None, embedder, tokenized_dir, output_dir, split, os.path.splitext(filename)[0]
            )
            for filename in file_list
        ]
        for future in tqdm(futures, desc=f"Processing {split} split", total=len(futures)):
            future.result()  # Wait for all futures to complete

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed tokenized data with LegalBERT.")
    parser.add_argument("--tokenized_dir", type=str, required=True, help="Path to tokenized data directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased", help="Model name for embedding.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (train/val/test).")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of examples to process.")

    args = parser.parse_args()
    embed_and_save(args.tokenized_dir, args.output_dir, args.model_name, args.split, args.sample_size)
