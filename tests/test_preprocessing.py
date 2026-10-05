"""Offline unit tests for the data pipeline (no downloads)."""
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from src.utils.dataloader import CorpusDataset, TedTalksDataset


def make_cfg(**preprocessing):
    return OmegaConf.create({
        "dataset": {"source": "en", "target": "es", "seed": 0, "test_size": 0.1, "val_size": 0.15,
                    "subset_fraction": 1.0},
        "preprocessing": {"lowercase": False, "remove_non_alphanumeric": False,
                          "remove_extra_whitespace": True, **preprocessing},
    })


def test_clean_text_keeps_accents_and_case_by_default():
    ds = CorpusDataset(make_cfg())
    assert ds.clean_text("  ¿Qué   pasó, Señor?  ") == "¿Qué pasó, Señor?"


def test_remove_non_alphanumeric_keeps_accented_letters():
    ds = CorpusDataset(make_cfg(remove_non_alphanumeric=True))
    assert ds.clean_text("¿Qué pasó, Señor?") == "Qué pasó Señor"


def test_collate_pads_inputs_with_pad_token_and_labels_with_ignore_index():
    ds = CorpusDataset(make_cfg())
    ds.pad_token_id = 1
    batch = ds.collate_fn([
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [8, 9]},
        {"input_ids": [5], "attention_mask": [1], "labels": [8, 9, 10]},
    ])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 1, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[8, 9, -100], [8, 9, 10]]


def test_split_carves_val_and_test_from_train_without_extra_subsetting():
    ds = CorpusDataset(make_cfg())
    train, val, test = ds.split_dataset(DatasetDict({"train": Dataset.from_dict({"x": list(range(1000))})}))
    assert (len(train), len(val), len(test)) == (750, 150, 100)


def test_split_uses_official_validation_and_test_splits():
    ds = CorpusDataset(make_cfg())
    splits = DatasetDict({name: Dataset.from_dict({"x": list(range(n))})
                          for name, n in (("train", 50), ("validation", 7), ("test", 9))})
    train, val, test = ds.split_dataset(splits)
    assert (len(train), len(val), len(test)) == (50, 7, 9)


def test_subset_fraction_is_applied_once_per_split():
    cfg = make_cfg()
    cfg.dataset.subset_fraction = 0.1
    train, val, test = CorpusDataset(cfg).split_dataset(
        DatasetDict({"train": Dataset.from_dict({"x": list(range(1000))})}))
    assert (len(train), len(val), len(test)) == (75, 15, 10)


def test_ted_talks_are_split_into_aligned_sentence_pairs():
    ds = TedTalksDataset(make_cfg())
    ds.tokenize = lambda dataset: dataset  # tokenization needs a downloaded tokenizer
    talks = Dataset.from_dict({"translation": [
        {"en": "Hello.\nThank you.", "es": "Hola.\nGracias."},
        {"en": "One line.", "es": "Una.\nDos."},  # misaligned, dropped
    ]})
    pairs = ds.prepare_split(talks)
    assert pairs["input_text"] == ["Hello.", "Thank you."]
    assert pairs["label_text"] == ["Hola.", "Gracias."]
