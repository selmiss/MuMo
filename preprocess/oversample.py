import pandas as pd
from datasets import concatenate_datasets


def sample_operation(tokenized_datasets, low_bound, high_bound):
    train_dataset = tokenized_datasets["train"]

    low_tail = train_dataset.filter(lambda x: x["labels"] <= low_bound)
    high_tail = train_dataset.filter(lambda x: x["labels"] >= high_bound)

    low_tail_oversampled = concatenate_datasets([low_tail] * 3)
    high_tail_oversampled = concatenate_datasets([high_tail] * 3)

    train_dataset_oversampled = concatenate_datasets(
        [train_dataset, low_tail_oversampled, high_tail_oversampled]
    )

    tokenized_datasets["train"] = train_dataset_oversampled

    return tokenized_datasets
