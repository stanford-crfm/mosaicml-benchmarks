"""
train.py

Run basic Hugging Face training of a causal language model

"""
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from transformers.data.data_collator import default_data_collator

from data import get_dataset

from paths import create_paths

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)

#from hf_flash_gpt import GPT2FlashLMHeadModel


@dataclass
class ModelArguments:
    """
    Arguments for the model
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from"
                " scratch."
            )
        },
    )

    model_config: Optional[str] = field(
        default=None,
        metadata={"help": ("The model config if training from scratch.")},
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataArguments:
    """
    Arguments for data
    """

    dataset_id: Optional[str] = field(
        default=None, metadata={"help": "Hugging Face dataset id."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face dataset name."},
    )

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory with text."}
    )

    validation_ratio: Optional[float] = field(
        default=0.0005, metadata={"help": "Ratio of data to use for validation set."}
    )

    seq_len: Optional[int] = field(
        default=1024, metadata={"help": "Sequence length of examples."}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=os.cpu_count(),  # use all available CPUs for processing
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


def train():
    # parse args
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set up paths
    hf_home = os.environ["HF_HOME"]
    run_id = training_args.run_name
    model_id = model_args.model_name_or_path
    run_dir = f"{hf_home}/runs/{run_id}"
    if not training_args.output_dir:
        training_args.output_dir = run_dir
    artifacts_dir = f"{hf_home}/artifacts"
    paths = create_paths(run_id, model_id, run_dir, artifacts_dir)

    # set up tokenizer
    tokenizer_path = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # set up model
    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        #model = AutoModelForCausalLM.from_pretrained(
            #model_args.model_name_or_path, config=config
        #)
        #model = GPT2FlashLMHeadModel.from_pretrained(
            #model_args.model_name_or_path, config=config
        #)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config
        )
    elif model_args.model_config:
        config = AutoConfig.from_pretrained(model_args.model_config)
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise RuntimeError(
            "One of model_name_or_path OR model_config must be specified!"
        )

    # set up dataset
    lm_dataset = get_dataset(
        tokenizer,
        paths,
        dataset_id=data_args.dataset_id,
        dataset_name=data_args.dataset_name,
        dataset_dir=data_args.dataset_dir,
        validation_ratio=data_args.validation_ratio,
        seq_len=data_args.seq_len,
        preprocessing_num_proc=data_args.preprocessing_num_workers,
    )

    #print("Training set length: ")
    #print(len(lm_dataset["train"][0]["input_ids"]))
    #print(lm_dataset["train"][0]["input_ids"])

    # set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # launch training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
    else:
        # this script assumes use of GPU if GPU's available
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

    if training_args.do_eval:
        metrics = trainer.evaluate()
        print(metrics)


if __name__ == "__main__":
    train()
