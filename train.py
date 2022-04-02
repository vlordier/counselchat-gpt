#!/usr/bin/env python

import os
import sys
import math
import random
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Dict, Union, Optional
import numpy as np
import pandas as pd
import torch

from datasets import load_metric, Dataset, DatasetDict

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    BatchEncoding,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import nltk
nltk.download('punkt')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    datafile_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None, metadata={"help":"Text column name in csv file"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    valid_percentage: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.datafile_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class DataCollatorForPromptGeneration:
    """
    Data collator used for line-by-line causal language modeling. Inputs are 
    dynamically padded to the maximum length of a batch if theyare not all of 
    the same length. The labels are constructed according to `toke_type_ids` 
    setting `label=-100` where `token_type_ids == 0` which corresponds to prompt. 

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of, max_length=self.max_length)
        
        labels = torch.where(batch["token_type_ids"].bool(), batch["input_ids"].clone(), torch.tensor(-100))
        batch["labels"] = labels
        return batch


def main():
    # parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # arguments can be passed in .json file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"""\
                Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change\n
                the `--output_dir` or add `--overwrite_output_dir` to train from scratch.
                """
            )
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Data preprocessing
    df = pd.read_csv(data_args.datafile_name)
    # standardise spaces
    df["questionTitle"] = df.questionTitle.map(lambda x: " ".join(x.split()))
    df["questionText"] = df.questionText.map(lambda x: " ".join(x.split()))
    df["answerText"] = df.answerText.map(lambda x: " ".join(x.split()))

    def mb_add_period(text):
        if text[-1] not in {"?", ".", "!"}:
            return text + "."
        return text

    df["questionTitle"] = df.questionTitle.map(mb_add_period)
    assert (df.questionTitle.str.endswith("?") | df.questionTitle.str.endswith(".") | df.questionTitle.str.endswith("!")).all()

    df["prompt"] = "Answer like a therapist:\n" + df.questionTitle + " " + df.questionText + "\nAnswer: "
    df["fullText"] = df.prompt + df.answerText

    df.rename(columns={"answerText":"answer"}, inplace=True)
    dataset = DatasetDict(**{
        k: Dataset.from_pandas(df.loc[df.split==k,["prompt", "answer", "topic"]]) for k in df.split.unique()
    })


    def tokenize(batch):
        return tokenizer(batch['prompt'], batch["answer"], return_token_type_ids=True, verbose=False, return_length=True, truncation=True, max_length=data_args.block_size)
    
    column_names = dataset["train"].column_names
    # add EOS tokens to each answer explicitly
    dataset = dataset.map(lambda x: {"answer":x["answer"]+tokenizer.eos_token}, batched=False)
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = tokenized_dataset["train"]
    valid_dataset = tokenized_dataset["val"]
    
    data_collator = DataCollatorForPromptGeneration(tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, use_cache= not training_args.gradient_checkpointing)

    rouge_metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Remove prompt from predictions and labels.
        predictions = np.where(labels != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        # Add mean generated length
        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        # result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(-1)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator, 
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer,
        # optimizers=(optimizer, None)
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    if training_args.do_eval:

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(valid_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    

if __name__ == "__main__":
    main()
