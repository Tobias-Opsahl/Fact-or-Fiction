from pathlib import Path

import pandas as pd
import torch
from constants import DATA_PATH, DATA_SPLIT_FILENAMES, FULL_FOLDER, SAVE_DATAFOLDER, SIMPLE_FOLDER, SUBGRAPH_FOLDER
from glocal_settings import SMALL, SMALL_SIZE
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils import get_logger, set_global_log_level

logger = get_logger(__name__)


def get_df(data_split, full=True):
    """
    Read and returns a dataframe of the dataset.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        full (bool, optional): Wether to use the full dataset or the simple one. Defaults to True.

    Raises:
        ValueError: If `data_split` is an unsuported string.

    Returns:
        pd.DataFrame: DataFrame of the dataset.
    """
    choices = ["train", "val", "test"]
    if data_split not in choices:
        raise ValueError(f"Argument `data_split` must be in {choices}. Was {data_split}. ")

    if full:
        intermediary_path = FULL_FOLDER
    else:
        intermediary_path = SIMPLE_FOLDER
    path = Path(DATA_PATH) / intermediary_path / DATA_SPLIT_FILENAMES[data_split]
    df = pd.read_csv(path)
    if SMALL:
        df = df[:SMALL_SIZE]
    return df


def get_subgraphs(data_split, subgraph_type):
    """
    Gets subgraph generated for dataset. Must be first found and saved in `retrieve_subgraph.py`.

   Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities) or `one_hop` (all of the entity edges).

    Raises:
        ValueError: If `data_split` is an unsuported string.

    Returns:
        pd.DataFrame: The dataframe of the subgraphs (as strings).
    """
    split_choices = ["train", "val", "test"]
    if data_split not in split_choices:
        raise ValueError(f"Argument `data_split` must be in {split_choices}. Was {data_split}. ")
    subgraph_choices = ["direct", "direct_filled", "one_hop"]
    if subgraph_type not in subgraph_choices:
        raise ValueError(f"Argument `subgraph_type` must be in {subgraph_choices}. Was {subgraph_type}. ")

    filename = "subgraphs_" + subgraph_type + "_" + data_split + ".csv"
    path = Path(SAVE_DATAFOLDER) / SUBGRAPH_FOLDER / filename
    df = pd.read_csv(path)
    if SMALL:
        df = df[:SMALL_SIZE]
    return df


class FactKGDataset(Dataset):
    def __init__(self, df, evidence=None):

        self.inputs = df["Sentence"]
        self.labels = df["Label"].astype(int)
        self.length = len(df)

        if evidence is not None:
            self.inputs = [self.inputs[i] + " | " + evidence["subgraph"][i] for i in range(self.length)]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return self.length


class CollateFunctor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input, labels = zip(*batch)
        labels = torch.tensor(labels)
        tokens = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        return tokens, labels


def get_dataloader(data_split, subgraph_type=None, model="bert-base-uncased", max_length=512, batch_size=64,
                   shuffle=True, drop_last=True):
    """
    Creates a dataloader for the desired data split and evidence (subgraph).

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities) or `one_hop` (all of the entity edges).
        model (str, optional): Name of model, in order to get tokenizer. Defaults to "bert-base-uncased".
        max_length (int, optional): Max tokenizer length. Defaults to 512.
        batch_size (int, optional): Batch size to dataloader. Defaults to 128.
        shuffle (bool, optional): Shuffle dataset. Defaults to True.
        drop_last (bool, optional): Drop last batch if it is less than `batch_size`. Defaults to True.

    Returns:
        DataLoader: The dataloader.
    """
    df = get_df(data_split, full=True)
    if subgraph_type is not None:
        subgraphs = get_subgraphs(data_split, subgraph_type)
    else:
        subgraphs = None
    dataset = FactKGDataset(df, subgraphs)
    tokenizer = AutoTokenizer.from_pretrained(model)
    collate_func = CollateFunctor(tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=collate_func)
    return dataloader


if __name__ == "__main__":
    set_global_log_level("debug")

    train_df = get_df("train", True)
    train_subgraphs = get_subgraphs("train", "direct_filled")
    model_name = 'bert-base-uncased'
    from IPython import embed
    embed()
