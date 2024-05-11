from pathlib import Path

import pandas as pd
import torch
from constants import DATA_PATH, DATA_SPLIT_FILENAMES, FULL_FOLDER, SAVE_DATAFOLDER, SIMPLE_FOLDER, SUBGRAPH_FOLDER
from glocal_settings import SMALL, SMALL_SIZE
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
from utils import get_logger, set_global_log_level, seed_everything

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


def get_embedding(text, tokenizer, model):
    """Generate an embedding for the input text using BERT base uncased model."""
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    # Use BERT to generate embeddings
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Get the embedding for the [CLS] token (first token)
    last_hidden_states = outputs.hidden_states[-1]  # The last layer hidden states
    cls_embedding = last_hidden_states[:, 0, :]
    # cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding


def convert_to_pyg_format(graph, tokenizer, model):
    if graph == []:
        return Data(x=None, edge_index=None, edge_attr=None)
    node_to_index = {}
    edge_to_index = {}
    node_features = []
    edge_features = []
    edge_indices = []

    # Assign indices to nodes and edges and create embeddings
    current_node_idx = 0
    current_edge_idx = 0
    for edge_list in graph:
        node1, edge, node2 = edge_list
        # Handle node1
        if node1 not in node_to_index:
            node_to_index[node1] = current_node_idx
            node_features.append(get_embedding(node1, tokenizer, model))
            current_node_idx += 1
        # Handle node2
        if node2 not in node_to_index:
            node_to_index[node2] = current_node_idx
            node_features.append(get_embedding(node2, tokenizer, model))
            current_node_idx += 1
        # Handle edge
        if edge not in edge_to_index:
            edge_to_index[edge] = current_edge_idx
            edge_features.append(get_embedding(edge, tokenizer, model))
            current_edge_idx += 1

        # Add to edge indices
        edge_indices.append([node_to_index[node1], node_to_index[node2]])

    # Convert lists to tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.stack(node_features)
    edge_attr = torch.stack(edge_features) if edge_features else None

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class FactKGDatasetGraph(Dataset):
    def __init__(self, df, evidence, tokenizer, model, max_length=512, mix_graphs=False):
        """
        Initialize the dataset. This dataset will return tokenized claims, graphs for the subgraph, and labels.

        Args:
            df (pd.DataFrame): FactKG dataframe
            evidence (pd.DataFram): Dataframe with the subgraphs, found by `retrieve_subgraphs.py`.
            tokenizer (transformer tokenizer): The Tokenizer for the model that embeds the graph.
            model (pytorch model): The model that embeds the graph.
            max_length (int, optional): Max length for the tokenizer. Defaults to 512.
            mix_graphs (bool, optional): If `True`, will use both the connected and the walkable graphs found in
                DBpedia. If `False`, will use connected if it is not empty, else walkable. Defaults to False.
        """
        self.inputs = df["Sentence"]
        self.labels = df["Label"].astype(int)
        self.length = len(df)
        self.subgraphs = evidence["walked"]
        self.evidence = evidence["subgraph"]
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.mix_graphs = mix_graphs

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.inputs[idx], return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_length)

        walked = self.subgraphs[idx]
        if self.mix_graphs:  # Use both connected and walkable graphs
            subgraph = walked["connected"]
            subgraph.extend(walked["walkable"])
        elif walked["connected"] != []:
            subgraph = walked["connected"]  # Use connected if it is not empty
        else:
            subgraph = walked["walkable"]  # Empty connceted, use walkable
        graph = convert_to_pyg_format(subgraph, tokenizer=self.tokenizer, model=self.model)

        label = torch.tensor(self.labels[idx])
        return tokens, graph, label

    def __len__(self):
        return self.length


def graph_collate_func(batch):
    token_batch, graph_batch, label_batch = zip(*batch)
    graph_batch = Batch.from_data_list(graph_batch)

    return token_batch, graph_batch, label_batch


def get_graph_dataloader(data_split, subgraph_type, model, model_name="bert-base-uncased", max_length=512,
                         batch_size=64, shuffle=True, drop_last=True):
    """
    Creates a dataloader for dataset with subgraph representation and tokenized text.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities) or `one_hop` (all of the entity edges).
        model (pytorch.Model): The actual model, to generate embeddings in the
        model_name (str, optional): Name of model, in order to get tokenizer. Defaults to "bert-base-uncased".
        max_length (int, optional): Max tokenizer length. Defaults to 512.
        batch_size (int, optional): Batch size to dataloader. Defaults to 128.
        shuffle (bool, optional): Shuffle dataset. Defaults to True.
        drop_last (bool, optional): Drop last batch if it is less than `batch_size`. Defaults to True.

    Returns:
        DataLoader: The dataloader.
    """
    df = get_df(data_split, full=True)
    subgraphs = get_subgraphs(data_split, subgraph_type)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = FactKGDatasetGraph(df, subgraphs, tokenizer, model, max_length)
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
