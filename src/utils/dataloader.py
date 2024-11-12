import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

class CorpusDataset(Dataset):
    def __init__(self, data: List[Dict], max_sentences_per_paragraph: int = 100, task: str = "unsupervised"):
        """
        Custom Dataset for handling NLP corpus for training models.
        
        Args:
            data (List[Dict]): List of data dictionaries containing 'text', 'input_ids', and 'attention_mask'.
            max_sentences_per_paragraph (int): Number of sentences to group together as a single data point (paragraph).
            task (str): Specifies if the task is 'supervised' or 'unsupervised'.
        """
        self.data = data
        self.max_sentences_per_paragraph = max_sentences_per_paragraph
        self.task = task
        self.paragraphs = self._create_paragraphs()

    def _create_paragraphs(self):
        """Groups sentences into paragraphs."""
        paragraphs = []
        for i in range(0, len(self.data), self.max_sentences_per_paragraph):
            if i + self.max_sentences_per_paragraph <= len(self.data):
                # Concatenate sentences to form a paragraph
                paragraph_texts = [entry["text"] for entry in self.data[i:i + self.max_sentences_per_paragraph]]
                paragraph_input_ids = [entry["input_ids"] for entry in self.data[i:i + self.max_sentences_per_paragraph]]
                paragraph_attention_masks = [entry["attention_mask"] for entry in self.data[i:i + self.max_sentences_per_paragraph]]

                # Flatten the lists of input_ids and attention_masks
                input_ids = [token for sublist in paragraph_input_ids for token in sublist]
                attention_mask = [mask for sublist in paragraph_attention_masks for mask in sublist]

                paragraphs.append({
                    "text": " ".join(paragraph_texts),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })
        return paragraphs

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        sample = self.paragraphs[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": sample["text"]
        }

        return result

def collate_fn(batch):
    """Custom collate function to handle variable-length input for DataLoader."""
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    texts = [item["text"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "texts": texts
    }

def get_data_loaders(tokenized_data, batch_size, max_sentences_per_paragraph=100, task="unsupervised"):
    dataset = CorpusDataset(data=tokenized_data, max_sentences_per_paragraph=max_sentences_per_paragraph, task=task)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    return data_loader
