# Some parts of codes are borrowed from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    # AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
)
from .roberta import RobertaForSequenceClassification_feat

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_raw_data(task_name, noise_ratio=0., cache_dir='./'):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    raw_datasets = load_dataset("glue", task_name)
    print(raw_datasets)

    # Labels
    if task_name is not None:
        is_regression = task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    if noise_ratio > 0.:
        raw_datasets = update_noise(raw_datasets, task_name, noise_ratio, cache_dir)

    return raw_datasets, label_list, num_labels, is_regression


def update_noise(raw_datasets, task_name, noise_ratio=0.1, cache_dir='./'):
    """Update dataset with noisy labels
    """
    path = os.path.join(cache_dir, f'{task_name}/target{noise_ratio}_large_4.pt')
    label = torch.load(path)['targets']
    raw_datasets = update_check_noise(raw_datasets, label)
    return raw_datasets


def update_check_noise(raw_datasets, label):
    """Check codes for update_noise
    """
    label_orig = torch.tensor(raw_datasets['train']['label'])
    raw_datasets['train'] = raw_datasets['train'].remove_columns("label").add_column("label", label)
    label_after = torch.tensor(raw_datasets['train']['label'])

    print(f"Noise label updated! {(label_orig == label_after).float().mean().item():.2f}")
    print((label_orig == 0).float().mean(), (label_after == 0).float().mean())
    print()
    return raw_datasets


def get_model(
    task_name,
    model_name,
    num_labels,
    ignore_mismatched_sizes=False,
    use_slow_tokenizer=False,
):
    """Load model and tokenizer
    """
    base_path = model_name
    if 'epoch' in base_path:
        base_path = '/'.join(base_path.split('/')[:-1])

    print(f"\nLoad config: {base_path}")
    config = AutoConfig.from_pretrained(base_path, num_labels=num_labels, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=not use_slow_tokenizer)
    model = RobertaForSequenceClassification_feat.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes)

    print(f"Loaded {model_name} {model.num_parameters()/(1024.**2):.0f} M parameters")
    return config, tokenizer, model


def get_processed_data(task_name,
                       model,
                       config,
                       tokenizer,
                       raw_datasets,
                       label_list,
                       num_labels,
                       is_regression=False,
                       pad_to_max_length=False,
                       max_length=512,
                       accelerator=None):
    """Preprocess data with tokenizer
    """
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and task_name is not None and not is_regression):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            print(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!")
            label_to_id = None
            # label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = ((examples[sentence1_key], ) if sentence2_key is None else
                 (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    if accelerator is not None:
        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    else:
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]

    return processed_datasets, train_dataset, eval_dataset
