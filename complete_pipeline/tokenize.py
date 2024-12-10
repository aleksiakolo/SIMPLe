import os
import json
import uuid
import re
import nltk
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

nltk.download('punkt')


class SentenceChunkTokenizer:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", max_input_length=512, preprocessing=None):
        """
        Initialize the tokenizer with efficient sentence chunking logic and optional preprocessing.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_length = max_input_length
        self.preprocessing = preprocessing or {}

    def preprocess_text(self, text):
        """
        Apply preprocessing steps to the text.
        """
        if self.preprocessing.get("lowercase", False):
            text = text.lower()
        if self.preprocessing.get("remove_non_alphanumeric", False):
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        if self.preprocessing.get("remove_extra_whitespace", False):
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_input(self, text):
        """
        Preprocess and split the input text into chunks that respect the token limit.
        """
        text = self.preprocess_text(text)
        sentences = nltk.tokenize.sent_tokenize(text)  # Split text into sentences

        # Precompute token lengths for all sentences
        tokenized_sentences = [self.tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
        token_lengths = [len(tokens) for tokens in tokenized_sentences]

        chunks = []
        current_chunk = []
        current_chunk_length = 0

        for i, (sentence, sentence_length) in enumerate(zip(sentences, token_lengths)):
            if current_chunk_length + sentence_length <= self.max_input_length:
                current_chunk.append(sentence)
                current_chunk_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_length = sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def tokenize_chunks(self, chunks):
        """
        Tokenize text chunks into input IDs and attention masks.
        """
        tokenized_chunks = self.tokenizer(
            chunks,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokenized_chunks

    def chunk_and_tokenize(self, text):
        """
        Full pipeline: preprocess, chunk, and tokenize the text.
        """
        chunks = self.preprocess_input(text)
        if not chunks:
            return []

        tokenized_chunks = self.tokenize_chunks(chunks)
        input_ids = tokenized_chunks["input_ids"].tolist()
        attention_mask = tokenized_chunks["attention_mask"].tolist()

        return [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(input_ids, attention_mask)]


def process_example(example, tokenizer, config, split, idx):
    """
    Process a single example by tokenizing its sources and summaries.
    """
    data = {}

    # Process sources
    sources = {}
    for i, source_text in enumerate(example[config.dataset.source_field]):
        tokenized_chunks = tokenizer.chunk_and_tokenize(source_text)
        source_data = {"text": source_text}
        for j, chunk in enumerate(tokenized_chunks):
            source_data[f"chunk_{j}"] = chunk
        sources[f"source_{i}"] = source_data

    data.update(sources)

    # Process summaries
    for summary_type in ["long", "small", "tiny"]:
        summary_text = example.get(f"summary/{summary_type}", None)
        if summary_text:
            tokenized_summary = tokenizer.chunk_and_tokenize(summary_text)
            data[summary_type] = summary_text
            input_ids = [chunk["input_ids"] for chunk in tokenized_summary]
            attention_masks = [chunk["attention_mask"] for chunk in tokenized_summary]
            data[f"{summary_type}_input_ids"] = input_ids
            data[f"{summary_type}_attention_mask"] = attention_masks
        else:
            data[summary_type] = None
            data[f"{summary_type}_input_ids"] = []
            data[f"{summary_type}_attention_mask"] = []

    return data


def save_data_point(data, output_dir, split, idx):
    """
    Save a single processed data point to a JSON file.
    """
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    output_path = os.path.join(output_dir, split, f"{idx}.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


@hydra.main(config_path="configs", config_name="lex_sum")
def tokenize_and_save_dataset(config: DictConfig):
    """
    Tokenize and save the dataset using multiprocessing.
    """
    logger.info("Starting dataset tokenization...")
    logger.info(OmegaConf.to_yaml(config))

    tokenizer = SentenceChunkTokenizer(
        model_name=config.tokenization.model_name,
        max_input_length=config.tokenization.max_length,
        preprocessing=config.get("preprocessing", {})
    )

    dataset = load_dataset(
        config.dataset.name, config.dataset.config_version, trust_remote_code=True
    )

    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split...")
        if config.sample_size:
            split_data = split_data.select(range(config.sample_size))

        os.makedirs(os.path.join(config.output.dir, split_name), exist_ok=True)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_example, example, tokenizer, config, split_name, idx)
                for idx, example in enumerate(split_data)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
                idx = futures[future]
                data_point = future.result()
                save_data_point(data_point, config.output.dir, split_name, idx)

    logger.info("Dataset tokenization completed.")


if __name__ == "__main__":
    tokenize_and_save_dataset()
