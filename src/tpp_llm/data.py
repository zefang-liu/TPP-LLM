"""
Dataset and Data Loader for the TPP-LLM
"""
import json
import os
import os.path
import random
import shutil

import torch
from torch.utils.data import Dataset

torch.manual_seed(0)


class TPPLLMDataset(Dataset):
    """
    TPP-LLM Dataset
    """

    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: dict) -> dict:
    """
    Data collation function for the data loader

    :param batch: items in a batch
    :return: batched data
    """
    return {
        'time_since_start': [torch.FloatTensor(item['time_since_start']) for item in batch],
        'time_since_last_event': [torch.FloatTensor(item['time_since_last_event']) for item in batch],
        'type_event': [torch.LongTensor(item['type_event']) for item in batch],
        'type_text': [item['type_text'] for item in batch]
    }


def create_few_shot_dataset(data_dir, output_dir, few_shot_ratio=0.1, seed=0) -> None:
    """
    Create the few-shot dataset

    :param data_dir: original data folder path
    :param output_dir: few-shot data folder path
    :param few_shot_ratio: few shot ratio for the training set
    :param seed: seed for reproducibility
    """
    random.seed(seed)

    train_file = os.path.join(data_dir, 'train.json')
    dev_file = os.path.join(data_dir, 'dev.json')
    test_file = os.path.join(data_dir, 'test.json')

    # Load train data
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    # Get the few-shot training data
    few_shot_size = int(len(train_data) * few_shot_ratio)
    few_shot_train_data = random.sample(train_data, few_shot_size)

    # Save the few-shot training data
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as file:
        json.dump(few_shot_train_data, file, indent=4)

    # Copy the dev and test files to the new directory
    shutil.copy(dev_file, os.path.join(output_dir, 'dev.json'))
    shutil.copy(test_file, os.path.join(output_dir, 'test.json'))

    print(f"Few-shot dataset created in {output_dir}.")
