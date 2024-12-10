import os
import json
from loguru import logger
from tqdm import tqdm


def validate_tokenized_data(data_dir, max_length=512):
    """
    Validate that all tokenized data chunks have a length less than or equal to the specified max_length.

    Args:
        data_dir (str): Path to the directory containing tokenized data.
        max_length (int): Maximum allowed token length for each chunk.

    Returns:
        None
    """
    logger.info(f"Starting validation for tokenized data in {data_dir} with max_length={max_length}")

    # Iterate through all files in the dataset
    for split in ["train", "validation", "test"]:
        # Initialize counters
        total_files = 0
        invalid_files = 0
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory {split_dir} does not exist. Skipping...")
            continue

        logger.info(f"Validating split: {split}")
        for file_name in tqdm(os.listdir(split_dir), desc=f"Validating {split}"):
            total_files += 1
            file_path = os.path.join(split_dir, file_name)

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Validate sources
                for source_key, source_data in data.items():
                    if source_key.startswith("source_"):
                        for chunk_key, chunk in source_data.items():
                            if chunk_key.startswith("chunk_"):
                                input_ids = chunk.get("input_ids", [])
                                if len(input_ids) > max_length:
                                    invalid_files += 1
                                    logger.error(
                                        f"Source chunk too long in file {file_name}, {source_key}-{chunk_key}: "
                                        f"{len(input_ids)} tokens"
                                    )

                # Validate summaries
                for summary_type in ["long", "small", "tiny"]:
                    input_ids_list = data.get(f"{summary_type}_input_ids", [])
                    for i, input_ids in enumerate(input_ids_list):
                        if len(input_ids) > max_length:
                            invalid_files += 1
                            logger.error(
                                f"Summary chunk too long in file {file_name}, {summary_type}-chunk_{i}: "
                                f"{len(input_ids)} tokens"
                            )

            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        # Summary
        logger.info(f"Validation completed for {split} split. Total files: {total_files}, Invalid files: {invalid_files}")


if __name__ == "__main__":
    tokenized_data_dir = "/home/aleksia/tokenized_dataset"  
    max_token_length = 512 
    validate_tokenized_data(tokenized_data_dir, max_length=max_token_length)
