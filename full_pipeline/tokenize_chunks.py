import os
import h5py
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
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
        embeddings = []
        with torch.no_grad():
            for chunk in tokenized_chunks:
                input_ids = torch.tensor(chunk["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(chunk["attention_mask"]).unsqueeze(0).to(self.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                chunk_embedding = output.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(chunk_embedding.squeeze(0).cpu())
        return torch.stack(embeddings)


def process_example(example_idx, embedder, tokenized_dir, output_dir, split):
    """
    Process a single example to generate embeddings and save them along with labels.
    """
    # Load tokenized JSON file
    tokenized_path = os.path.join(tokenized_dir, split, f"{example_idx}.json")
    if not os.path.exists(tokenized_path):
        logger.error(f"Tokenized file {tokenized_path} does not exist.")
        return

    with open(tokenized_path, "r") as f:
        tokenized_data = json.load(f)

    # Create HDF5 file for embeddings
    h5_path = os.path.join(output_dir, "data", split, f"{example_idx}.h5")
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
                h5_file.create_dataset(source_key, data=embeddings.numpy())

    # Save labels (target summaries and tokens) as JSON
    labels = {}
    for summary_type in ["long", "small", "tiny"]:
        if summary_type in tokenized_data:
            labels[summary_type] = tokenized_data[summary_type]
            labels[f"{summary_type}_input_ids"] = tokenized_data.get(f"{summary_type}_input_ids", [])
            labels[f"{summary_type}_attention_mask"] = tokenized_data.get(f"{summary_type}_attention_mask", [])

    labels_path = os.path.join(output_dir, "labels", split, f"{example_idx}.json")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)


@hydra.main(version_base=None, config_path="./configs", config_name="embed_lex_sum")
def embed_and_save(config: DictConfig):
    """
    Embed tokenized sources and save embeddings (HDF5) and labels (JSON).
    """
    logger.info("Starting embedding process...")
    logger.info(OmegaConf.to_yaml(config))

    embedder = LegalBERTEmbedder(model_name=config.model.name, device=config.model.device)
    tokenized_dir = config.paths.tokenized_dir
    output_dir = config.paths.output_dir
    split = config.split
    sample_size = config.sample_size

    tokenized_split_dir = os.path.join(tokenized_dir, split)

    # Get list of tokenized files
    file_list = [f for f in os.listdir(tokenized_split_dir) if f.endswith(".json")]
    if sample_size:
        file_list = file_list[:sample_size]

    logger.info(f"Processing {len(file_list)} examples in the {split} split...")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_example, os.path.splitext(filename)[0], embedder, tokenized_dir, output_dir, split): filename
            for filename in file_list
        }
        for future in tqdm(futures, desc=f"Processing {split} split", total=len(futures)):
            try:
                future.result()  # Wait for all futures to complete
            except Exception as e:
                logger.error(f"Error processing example {futures[future]}: {e}")

    logger.info("Embedding process completed.")


if __name__ == "__main__":
    embed_and_save()
