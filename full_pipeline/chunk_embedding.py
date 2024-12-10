import os
import h5py
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import hydra


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
        if not tokenized_chunks:
            logger.error("No valid tokenized chunks provided to embed_chunks.")
            return torch.empty(0)  # Return empty tensor if no chunks
        embeddings = []
        with torch.no_grad():
            for chunk in tokenized_chunks:
                input_ids = torch.tensor(chunk.get("input_ids", [])).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(chunk.get("attention_mask", [])).unsqueeze(0).to(self.device)
                if input_ids.numel() == 0 or attention_mask.numel() == 0:
                    logger.warning(f"Empty input_ids or attention_mask for chunk: {chunk}")
                    continue  # Skip invalid chunks
                
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                chunk_embedding = output.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(chunk_embedding.squeeze(0).cpu())

        if not embeddings:
            logger.warning("No embeddings generated from the provided chunks.")
            return torch.empty(0)  # Return empty tensor if no embeddings

        return torch.stack(embeddings)


def process_single_file(file_name, embedder, config):
    """
    Process a single example.
    """
    tokenized_dir = config.paths.tokenized_dir
    output_dir = config.paths.output_dir
    split = config.split

    tokenized_path = os.path.join(tokenized_dir, split, file_name)
    if not os.path.exists(tokenized_path):
        logger.error(f"Tokenized file {tokenized_path} does not exist.")
        return

    with open(tokenized_path, "r") as f:
        tokenized_data = json.load(f)

    # Create HDF5 file for embeddings
    h5_path = os.path.join(output_dir, "data", split, file_name.replace(".json", ".h5"))
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as h5_file:
        for source_key, source_data in tokenized_data.items():
            if source_key.startswith("source_"):
                chunks = [
                    source_data[f"chunk_{j}"]
                    for j in range(len(source_data))
                    if f"chunk_{j}" in source_data
                ]
                embeddings = embedder.embed_chunks(chunks)
                if embeddings.nelement() > 0:  # Only save non-empty embeddings
                    h5_file.create_dataset(source_key, data=embeddings.numpy())
                    logger.info(f"Saved embeddings for {source_key} in {file_name}")
                else:
                    logger.warning(f"No embeddings generated for {source_key} in {file_name}")

    # Save labels (only the long summary) as JSON
    labels = {}
    if "long" in tokenized_data:
        labels["long"] = {
            "text": tokenized_data["long"]["text"],
            "chunks": {
                key: tokenized_data["long"][key]
                for key in tokenized_data["long"].keys()
                if key.startswith("chunk_")
            },
        }

    labels_path = os.path.join(output_dir, "labels", split, file_name)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)
    logger.info(f"Saved labels for file: {file_name}")


@hydra.main(version_base=None, config_path="./configs", config_name="embed_lex_sum")
def embed_and_save(config: DictConfig):
    """
    Embed tokenized sources and save embeddings (HDF5) and labels (JSON).
    """
    logger.info("Starting embedding process...")
    logger.info(OmegaConf.to_yaml(config))

    tokenized_dir = config.paths.tokenized_dir
    output_dir = config.paths.output_dir
    split = config.split

    tokenized_split_dir = os.path.join(tokenized_dir, split)

    # Get list of tokenized files
    file_list = [f for f in os.listdir(tokenized_split_dir) if f.endswith(".json")]
    if config.sample_size:
        file_list = file_list[:config.sample_size]

    logger.info(f"Processing {len(file_list)} examples in the {split} split...")

    embedder = LegalBERTEmbedder(
        model_name=config.model.name,
        device=config.model.device,
    )

    # Process files one by one
    for file_name in tqdm(file_list, desc=f"Processing {split} split"):
        process_single_file(file_name, embedder, config)
        torch.cuda.empty_cache()  # Clear GPU memory after each file

    logger.info("Embedding process completed.")


if __name__ == "__main__":
    embed_and_save()