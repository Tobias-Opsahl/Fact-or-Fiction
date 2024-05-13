import pickle
from pathlib import Path

import pandas as pd
import torch
from constants import (BERT_LAST_LAYER_DIM, DATA_PATH, DATA_SPLIT_FILENAMES, EMBEDDINGS_FILENAME, FULL_FOLDER,
                       SAVE_DATAFOLDER, SIMPLE_FOLDER, SUBGRAPH_FOLDER)
from glocal_settings import SMALL, SMALL_SIZE
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer
from utils import get_logger, seed_everything, set_global_log_level

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

    filename = "subgraphs_" + subgraph_type + "_" + data_split + ".pkl"
    path = Path(SAVE_DATAFOLDER) / SUBGRAPH_FOLDER / filename
    df = pd.read_pickle(path)
    if SMALL:
        df = df[:SMALL_SIZE]
    return df


def get_precomputed_embeddings():
    """
    Gets dict with precomputed embeddings, made with `make_subgraph_embeddings.py`.

    Raises:
        ValueError: If `data_split` is an unsuported string.

    Returns:
        dict: The dict of the subgraphs (as strings).
    """
    path = Path(SAVE_DATAFOLDER) / EMBEDDINGS_FILENAME
    embedding_dict = pickle.load(open(path, "rb"))
    return embedding_dict


class FactKGDataset(Dataset):
    def __init__(self, df, evidence=None):
        """
        Args:
            df (pd.DataFrame): Dataframe with claims ("Sentence") and labels ("Label").
            evidence (list, optional): List of the subgraph evidences to use, will be converted to string. `None`
                if no evidence should be used.
        """

        self.inputs = df["Sentence"]
        self.labels = df["Label"].astype(int)
        self.length = len(df)

        if evidence is not None:
            self.inputs = [self.inputs[i] + " | " + str(evidence[i]) for i in range(self.length)]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return self.length


class CollateFunctor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        inputs, labels = zip(*batch)
        labels = torch.tensor(labels)
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs["labels"] = torch.as_tensor(labels)
        return inputs


def get_dataloader(data_split, subgraph_type=None, subgraph_to_use="discovered", model="bert-base-uncased",
                   max_length=512, batch_size=64, shuffle=True, drop_last=True):
    """
    Creates a dataloader for the desired data split and evidence (subgraph).

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities) or `one_hop` (all of the entity edges).
        subgraph_to_use (str). In ["discovered", "connected", "walkable"]. "discovered" means that we use the string
            representation of what directly found with `subgraph_type`. "Connected" means that walk the nodes and
            relations found with `subgraph_type`, and use the connected graphs if found, and the walkable if not.
            "walkable" means we use both the connected graphs and the walkable graphs.
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
        choices = ["discovered", "connected", "walkable"]
        if subgraph_to_use not in choices:
            raise ValueError(f"Argument `subgraph_to_use` must be in {choices}. Was {subgraph_to_use}. ")
        if subgraph_to_use == "discovered":
            evidence = subgraphs["subgraph"]
        elif subgraph_to_use == "connected":
            evidence = []
            for i in range(len(subgraphs)):
                if subgraphs["walked"][i]["connected"] == []:
                    evidence.append(subgraphs["walked"][i]["connected"])
                else:
                    evidence.append(subgraphs["walked"][i]["walkable"])
        elif subgraph_to_use == "walkable":
            evidence = []
            for i in range(len(subgraphs)):
                evidence.append(subgraphs["walked"][i]["connected"] + subgraphs["walked"][i]["walkable"])
    else:
        evidence = None

    dataset = FactKGDataset(df, evidence)
    tokenizer = AutoTokenizer.from_pretrained(model)
    collate_func = CollateFunctor(tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=collate_func)
    return dataloader


def get_embedding(text, embeddings_dict):
    if embeddings_dict.get(text) is not None:
        return torch.tensor(embeddings_dict[text])
    return torch.zeros(BERT_LAST_LAYER_DIM)


def convert_to_pyg_format(graph, embedding_dict):
    # if graph == []:
    #     return Data(x=[], edge_index=[], edge_attr=[])
    node_to_index = {}  # Node text to int mapping
    edge_to_index = {}  # Same for edges
    node_features = []  # List of embeddings
    edge_features = []  # Same for edges
    edge_indices = []

    current_node_idx = 0
    current_edge_idx = 0
    for edge_list in graph:
        node1, edge, node2 = edge_list  # Graph consists of list on the format [node1, edge, node2]

        if node1 not in node_to_index:
            node_to_index[node1] = current_node_idx
            node_features.append(get_embedding(node1, embedding_dict))
            current_node_idx += 1

        if node2 not in node_to_index:
            node_to_index[node2] = current_node_idx
            node_features.append(get_embedding(node2, embedding_dict))
            current_node_idx += 1

        if edge not in edge_to_index:
            edge_to_index[edge] = current_edge_idx
            edge_features.append(get_embedding(edge, embedding_dict))
            current_edge_idx += 1

        edge_indices.append([node_to_index[node1], node_to_index[node2]])

    edge_index = torch.tensor(edge_indices).t().contiguous()  # Transpose and make memory contigious
    x = torch.stack(node_features)
    edge_attr = torch.stack(edge_features)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class FactKGDatasetGraph(Dataset):
    def __init__(self, df, evidence, embedding_dict, mix_graphs=False):
        """
        Initialize the dataset. This dataset will return tokenized claims, graphs for the subgraph, and labels.

        Args:
            df (pd.DataFrame): FactKG dataframe
            evidence (pd.DataFram): Dataframe with the subgraphs, found by `retrieve_subgraphs.py`.
            embedding_dict (dict): Dictionary mapping the knowledge graph words to embeddings.
            mix_graphs (bool, optional): If `True`, will use both the connected and the walkable graphs found in
                DBpedia. If `False`, will use connected if it is not empty, else walkable. Defaults to False.
        """
        self.inputs = df["Sentence"]
        self.labels = df["Label"].astype(int)
        self.length = len(df)
        self.subgraphs = evidence["walked"]
        self.evidence = evidence["subgraph"]
        self.embedding_dict = embedding_dict
        self.mix_graphs = mix_graphs

    def __getitem__(self, idx):
        claims = self.inputs[idx]

        walked = self.subgraphs[idx]
        if self.mix_graphs:  # Use both connected and walkable graphs
            subgraph = walked["connected"]
            subgraph.extend(walked["walkable"])
        elif walked["connected"] != []:
            subgraph = walked["connected"]  # Use connected if it is not empty
        else:
            subgraph = walked["walkable"]  # Empty connceted, use walkable
        graph = convert_to_pyg_format(subgraph, self.embedding_dict)

        label = self.labels[idx]
        return claims, graph, label

    def __len__(self):
        return self.length


class GraphCollateFunc:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        inputs, graph_batch, labels = zip(*batch)
        tokens = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_length)
        graph_batch = Batch.from_data_list(graph_batch)
        labels = torch.tensor(labels).float()

        return tokens, graph_batch, labels


def get_graph_dataloader(data_split, subgraph_type, bert_model_name="bert-base-uncased", max_length=512,
                         batch_size=64, shuffle=True, drop_last=True, mix_graphs=False):
    """
    Creates a dataloader for dataset with subgraph representation and tokenized text.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities) or `one_hop` (all of the entity edges).
        bert_model_name (str, optional): Name of model, in order to get tokenizer. Defaults to "bert-base-uncased".
        max_length (int, optional): Max tokenizer length. Defaults to 512.
        batch_size (int, optional): Batch size to dataloader. Defaults to 128.
        shuffle (bool, optional): Shuffle dataset. Defaults to True.
        drop_last (bool, optional): Drop last batch if it is less than `batch_size`. Defaults to True.
        mix_graphs (bool, optional): If `True`, will use both the connected and the walkable graphs found in
                DBpedia. If `False`, will use connected if it is not empty, else walkable. Defaults to False.

    Returns:
        DataLoader: The dataloader.
    """
    df = get_df(data_split, full=True)
    subgraphs = get_subgraphs(data_split, subgraph_type)
    embedding_dict = get_precomputed_embeddings()

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    graph_collate_func = GraphCollateFunc(tokenizer, max_length=max_length)
    dataset = FactKGDatasetGraph(
        df, subgraphs, embedding_dict=embedding_dict, mix_graphs=mix_graphs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=graph_collate_func)
    return dataloader


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(57)

    from models import get_bert_model
    model = get_bert_model("bert")
    dataloader = get_graph_dataloader("val", "direct_filled", model, batch_size=3)
    a = next(iter(dataloader))
    from IPython import embed
    embed()
