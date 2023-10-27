"""
data.py

Build a dataset from .jsonl containing text examples

"""
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import datasets
from transformers import BatchEncoding, PreTrainedTokenizer


def get_dataset(
    tokenizer: PreTrainedTokenizer,
    paths: Dict[str, Path],
    dataset_id: str,
    dataset_name: str,
    dataset_dir: str = None,
    validation_ratio: float = 0.0005,
    seq_len: int = 1024,
    preprocessing_num_proc: int = 64,
    stride: int = -1,
    ignore_train: bool = False,
) -> datasets.DatasetDict:
    """Run basic tokenization and grouping to turn a Hugging Face Dataset (via `datasets`) into a torch.Dataset."""

    # Sanity check on input args
    stride = seq_len if stride < 0 else stride
    assert (
        stride <= seq_len
    ), f"Data grouping stride ({stride}) is smaller than sequence length: we are losing data."
    if dataset_dir is not None:
        file_names = os.listdir(dataset_dir)
        file_type = os.path.splitext(file_names[0])[1][1:]
        dataset_files = {}
        dataset_files["train"] = [
            f"{dataset_dir}/{fn}"
            for fn in file_names
            if "train" in fn and fn.endswith(file_type)
        ]
        dataset_files["validation"] = [
            f"{dataset_dir}/{fn}"
            for fn in file_names
            if "validation" in fn and fn.endswith(file_type)
        ]
        file_type = "json" if file_type == "jsonl" else file_type
        assert file_type in ["json", "txt", "csv"]
        dataset = datasets.load_dataset(
            file_type,
            name=dataset_name,
            data_files=dataset_files,
            cache_dir=str(paths["dataset"]),
        )
    else:
        dataset = datasets.load_dataset(
            dataset_id, name=dataset_name, cache_dir=str(paths["dataset"])
        )

    if "validation" not in dataset:
        assert (
            "train" in dataset
        ), "You must have train in dataset to make a validation dataset"
        # Create Dataset Split Cache Files
        train_fn, val_fn = [
            str(paths["dataset"] / dataset_id / f"{k}-split.hf")
            for k in ["train", "val"]
        ]
        dataset = dataset["train"].train_test_split(
            test_size=validation_ratio,
            train_indices_cache_file_name=train_fn,
            test_indices_cache_file_name=val_fn,
        )
        dataset["validation"] = dataset["test"]
        del dataset["test"]

    # Preprocess Dataset in a Streaming Fashion
    assert "train" in dataset, "Field `train` not in Dataset!"
    if ignore_train:
        del dataset["train"]
        assert (
            len(dataset) > 0
        ), "You can't set ignore_train = True when there is only train data"

    # Second, run straight-up tokenization
    def tokenize(examples: Dict[str, List[str]]) -> BatchEncoding:
        pretokenize_examples = [x + tokenizer.eos_token for x in examples["text"]]
        return tokenizer(pretokenize_examples)

    # Create Post-Tokenization Cache Paths
    post_tokenization_cache_files = {
        k: str(
            paths["preprocessed"]
            / dataset_id
            / "preprocessing"
            / "tokenization"
            / f"{k}-tokenized.hf"
        )
        for k in dataset
    }
    # Create Parent Path of Cache Files
    (paths["preprocessed"] / dataset_id / "preprocessing" / "tokenization").mkdir(
        parents=True, exist_ok=True
    )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=preprocessing_num_proc,
        remove_columns=next(iter(dataset.values())).column_names,
        cache_file_names=post_tokenization_cache_files,
        load_from_cache_file=True,
    )

    # Finally, actually run chunking (collapse multiple sequences into a giant document to read `seq_len` chunks from)
    def group(examples: Dict[str, Iterable[List[int]]]) -> Dict[str, List[List[int]]]:
        # Concatenate all the Texts
        concatenated: Dict[str, List[int]] = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the "very last" bit of the dataset that doesn't fit into block size...
        total_length = ((total_length - seq_len + stride) // stride) * stride

        # Split by Chunks of Maximum Length
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, stride)]
            for k, t in concatenated.items()
        }
        result["labels"] = deepcopy(result["input_ids"])

        # Mask out losses in overlapping regions. If training data, string will be equal to seq_len
        for i, labels in enumerate(result["labels"]):
            if i == 0:
                continue
            for j in range(len(labels) - stride):
                labels[j] = -100
            result["labels"][i] = labels
        return result

    # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
    # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
    # value might be slower to preprocess.

    # Create Post-Chunking Cache Paths
    post_chunking_cache_files = {
        k: str(
            paths["preprocessed"]
            / dataset_id
            / "preprocessing"
            / "chunking"
            / f"{k}-stride={stride}-chunked.hf"
        )
        for k in dataset
    }
    # Create Parent Path of Cache Files
    (paths["preprocessed"] / dataset_id / "preprocessing" / "chunking").mkdir(
        parents=True, exist_ok=True
    )

    lm_dataset = tokenized_dataset.map(
        group,
        batched=True,
        num_proc=preprocessing_num_proc,
        cache_file_names=post_chunking_cache_files,
        load_from_cache_file=True,
    )

    return lm_dataset

