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


def get_df(data_split, full=True, small=None):
    """
    Read and returns a dataframe of the dataset.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        full (bool, optional): Wether to use the full dataset or the simple one. Defaults to True.
        small (bool): Pass to override `SMALL`.

    Raises:
        ValueError: If `data_split` is an unsuported string.

    Returns:
        pd.DataFrame: DataFrame of the dataset.
    """
    choices = ["train", "val", "test"]
    if data_split not in choices:
        raise ValueError(f"Argument `data_split` must be in {choices}. Was {data_split}. ")

    if small is None:
        small = SMALL

    if full:
        intermediary_path = FULL_FOLDER
    else:
        intermediary_path = SIMPLE_FOLDER
    path = Path(DATA_PATH) / intermediary_path / DATA_SPLIT_FILENAMES[data_split]
    df = pd.read_csv(path)
    if small:
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
    subgraph_choices = ["direct", "direct_filled", "one_hop", "relevant"]
    if subgraph_type not in subgraph_choices:
        raise ValueError(f"Argument `subgraph_type` must be in {subgraph_choices}. Was {subgraph_type}. ")

    filename = "subgraphs_" + subgraph_type + "_" + data_split + ".pkl"
    path = Path(SAVE_DATAFOLDER) / SUBGRAPH_FOLDER / filename
    df = pd.read_pickle(path)
    if SMALL:
        df = df[:SMALL_SIZE]
    return df


def calculate_embeddings(text, tokenizer, model, with_classifier=True):
    """
    Calculate embeddings for text, given a tokenizer and a model

    Args:
        text (list of str): List of the strings to make embeddings for.
        tokenizer (tokenizer): The tokenizer.
        model (pytorch model): The model.
        with_classifier (bool): If model has classifier (should reach hidden state) or not (output is hidden state).

    Returns:
        dict: Dict mapping from text to embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move to device

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embedding for the [CLS] token (first token)
    if with_classifier:
        last_hidden_states = outputs.hidden_states[-1]
    else:
        last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[:, 0, :]
    return cls_embedding


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


def get_embedding(text, online_embeddings, embeddings_dict, tokenizer, model):
    if online_embeddings:
        return calculate_embeddings(text, tokenizer, model, with_classifier=False).squeeze()

    if embeddings_dict.get(text) is not None:
        return torch.tensor(embeddings_dict[text])
    return torch.zeros(BERT_LAST_LAYER_DIM)


def convert_to_pyg_format(graph, online_embeddings, embedding_dict=None, tokenizer=None, model=None):
    """
    Convert graph on DBpedia dict format to torch_embedding.data format, so it can be run in GNN.

    Args:
        graph (dict): Dict of graph, gotten by calling `kg.search()` on each element in the graph.
        online_embeddings (bool): If True, will calculate embeddings for knowledge subgraph online, with a model
                that might be tuned during the training.
        embedding_dict (dict): Dict mapping words to embeddings, to be used as node and edge features.
            This should be precomputed.
        tokenizer (tokenizer): Tokenizer to `model` if `online_embeddings`.
        model (pytroch model): Model to compute embeddings if `online_embeddings`.

    Returns:
        torch_geometric.data: Graph data.
    """
    if online_embeddings and (model is None or tokenizer is None):
        message = "Argument `model` or `tokenizer` can not be `None` when `online_embeddings` is True."
        raise ValueError(message)
    if not online_embeddings and embedding_dict is None:
        message = "Argument `embedding_dict` can not be `None` when `online_embeddings` is False."
        raise ValueError(message)

    if graph == []:  # Dummy empty graph. Not actually empty because of vectorized computations.
        graph = [["none", "none", "none"]]
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
            embedding = get_embedding(node1, online_embeddings=online_embeddings, embeddings_dict=embedding_dict,
                                      tokenizer=tokenizer, model=model)
            node_features.append(embedding)
            current_node_idx += 1

        if node2 not in node_to_index:
            node_to_index[node2] = current_node_idx
            embedding = get_embedding(node2, online_embeddings=online_embeddings, embeddings_dict=embedding_dict,
                                      tokenizer=tokenizer, model=model)
            node_features.append(embedding)
            current_node_idx += 1

        if edge not in edge_to_index:
            edge_to_index[edge] = current_edge_idx
            embedding = get_embedding(edge, online_embeddings=online_embeddings, embeddings_dict=embedding_dict,
                                      tokenizer=tokenizer, model=model)
            edge_features.append(embedding)
            current_edge_idx += 1

        edge_indices.append([node_to_index[node1], node_to_index[node2]])

    edge_index = torch.tensor(edge_indices).t().contiguous()  # Transpose and make memory contigious
    x = torch.stack(node_features)
    edge_attr = torch.stack(edge_features)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class FactKGDatasetGraph(Dataset):
    def __init__(self, df, evidence, online_embeddings=False, embedding_dict=None, tokenizer=None, model=None,
                 mix_graphs=False):
        """
        Initialize the dataset. This dataset will return tokenized claims, graphs for the subgraph, and labels.

        Args:
            df (pd.DataFrame): FactKG dataframe
            evidence (pd.DataFram): Dataframe with the subgraphs, found by `retrieve_subgraphs.py`.
            online_embeddings (bool): If True, will calculate embeddings for knowledge subgraph online, with a model
                that might be tuned during the training.
            embedding_dict (dict): Dict mapping the knowledge graph words to embeddings if not `online_embeddings`.
            tokenizer (tokenizer): Tokenizer to `model` if `online_embeddings`.
            model (pytroch model): Model to compute embeddings if `online_embeddings`.
            mix_graphs (bool, optional): If `True`, will use both the connected and the walkable graphs found in
                DBpedia. If `False`, will use connected if it is not empty, else walkable. Defaults to False.
        """
        if online_embeddings and (model is None or tokenizer is None):
            message = "Argument `model` or `tokenizer` can not be `None` when `online_embeddings` is True."
            raise ValueError(message)
        if not online_embeddings and embedding_dict is None:
            message = "Argument `embedding_dict` can not be `None` when `online_embeddings` is False."
            raise ValueError(message)
        self.inputs = df["Sentence"]
        self.labels = df["Label"].astype(int)
        self.length = len(df)
        self.subgraphs = evidence["walked"]
        self.evidence = evidence["subgraph"]
        self.online_embeddings = online_embeddings
        self.embedding_dict = embedding_dict
        self.tokenizer = tokenizer
        self.model = model
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
        graph = convert_to_pyg_format(
            subgraph, online_embeddings=self.online_embeddings, embedding_dict=self.embedding_dict,
            tokenizer=self.tokenizer, model=self.model)

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


def get_graph_dataloader(
        data_split, subgraph_type, online_embeddings=False, model=None, bert_model_name="bert-base-uncased",
        max_length=512, batch_size=64, shuffle=True, drop_last=True, mix_graphs=False):
    """
    Creates a dataloader for dataset with subgraph representation and tokenized text.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities), `one_hop` (all of the entity edges) or `relevant` (direct plus
            edges that appears in claim).
        online_embeddings (bool): If True, will calculate embeddings for knowledge subgraph online, with a model
                that might be tuned during the training.
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

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    graph_collate_func = GraphCollateFunc(tokenizer, max_length=max_length)
    if online_embeddings:
        dataset = FactKGDatasetGraph(
            df, subgraphs, online_embeddings=online_embeddings, embedding_dict=None,
            tokenizer=tokenizer, model=model, mix_graphs=mix_graphs)
    else:
        embedding_dict = get_precomputed_embeddings()
        dataset = FactKGDatasetGraph(
            df, subgraphs, online_embeddings=online_embeddings, embedding_dict=embedding_dict, mix_graphs=mix_graphs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=graph_collate_func)
    return dataloader
