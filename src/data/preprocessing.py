import os
import json
import random
from typing import List, Dict, Tuple

def extract_all_labels(data_dir: str) -> List[str]:
    """
    Parses all JSON files to extract unique operation labels (e.g., extrude1, fillet2).
    Returns a sorted list of unique labels.
    """
    label_set = set()
    for fname in os.listdir(data_dir):
        if fname.endswith('.json'):
            with open(os.path.join(data_dir, fname), 'r') as f:
                sample = json.load(f)
                for face in sample['faces']:
                    for label in face['labels']:
                        label_set.add(label)
    return sorted(label_set)

def build_label_map(labels: List[str]) -> Dict[str, int]:
    """
    Maps each label (operation name) to a unique integer index.
    """
    return {label: idx for idx, label in enumerate(labels)}

def split_dataset(data_dir: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly splits dataset filenames into train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    random.shuffle(filenames)

    total = len(filenames)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = filenames[:train_end]
    val_files = filenames[train_end:val_end]
    test_files = filenames[val_end:]

    return train_files, val_files, test_files

def save_splits(split_dir: str, train: List[str], val: List[str], test: List[str]):
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.writelines(line + '\n' for line in train)
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.writelines(line + '\n' for line in val)
    with open(os.path.join(split_dir, 'test.txt'), 'w') as f:
        f.writelines(line + '\n' for line in test)
