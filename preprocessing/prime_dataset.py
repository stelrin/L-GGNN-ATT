from typing import NamedTuple
import numpy as np
import csv

class DatasetMetadata(NamedTuple):
    name: str
    max_number_of_nodes: int
    max_sequence_len: int
    item_count: int
    train_dataset_size: int
    test_dataset_size: int


def get_dataset_metadata(dataset_directory: str):
    item_count = 0
    max_sequence_len = 0
    max_number_of_nodes = 0
    train_dataset_size = 0
    test_dataset_size = 0

    for dataset_type in ["train", "test"]:
        with open(f"datasets/{dataset_directory}/{dataset_type}.csv", "r") as data_file:
            data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=",")]
            item_count = max(item_count, np.amax([np.amax(z) for z in data]) + 1)  # (+ 1) for that extra 0 as the item ids start at 1
            max_sequence_len = max(max_sequence_len, len(max(data, key=len)))
            max_number_of_nodes = max(max_number_of_nodes, len(max([np.unique(i) for i in data], key=len)))

            if dataset_type == "train":
                train_dataset_size = len(data)
            else:
                test_dataset_size = len(data)

    return DatasetMetadata(
        dataset_directory,
        max_number_of_nodes,
        max_sequence_len,
        item_count,
        train_dataset_size,
        test_dataset_size,
    )
