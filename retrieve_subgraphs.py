import argparse
import ast
import csv
import os
import pickle
from pathlib import Path

import pandas as pd
from constants import (DATA_PATH, DBPEDIA_LIGHT_FILENAME, DPBEDIA_FOLDER, FULL_FOLDER, SAVE_DATAFOLDER, SUBGRAPH_FOLDER,
                       TEST_FILENAME, TRAIN_FILENAME, VAL_FILENAME)
from glocal_settings import LOCAL
from utils import get_logger, set_global_log_level

logger = get_logger(__name__)


class KnowledgeGraph:
    def __init__(self, kg):
        self.kg = kg

    def get_direct_subgraph(self, entity_list, fill_if_empty=False):
        """
        Finds the direct subgraph, which means the graph with `entity_list` as nodes, and all the direct relations
        between them as edges.
        If `fill_if_empty` is True, will use the one-hop (include every relation for the entities) if there was no
        edges added.

        Args:
            entity_list (list of str): List of the entity names.
            fill_if_empty (bool, optional): If True, will fill the nodes with `one-hop` (all the relations for the
                entities are included) if the graph gets no edges. Defaults to False.

        Returns:
            dict: Dict representing the graph, om the same format as with FactKG.
        """
        subgraph = {}
        empty = True

        for entity in entity_list:
            subgraph[entity] = []
            node = self.kg.get(entity)
            if node is None:
                logger.warn(f"Argument {entity} was in `entity_list`, but not in knowledge graph. ")
                continue
            for relation, neighbours in self.kg[entity].items():
                # `neighbours` is a list of nodes `entity` point to with a `relation`-edge.
                for neighbour in neighbours:
                    if neighbour in entity_list:
                        empty = False
                        subgraph[entity].append(relation)

        if empty and fill_if_empty:  # No relations are filled, so we will include the full entity nodes
            subgraph = self.get_one_hop_subgraph(entity_list)
        return subgraph

    def get_one_hop_subgraph(self, entity_list):
        """
        Includes all relations for every entity in `entity_list` as a subgraph.

        Args:
            entity_list (list of str): List of the entity names.

        Returns:
            dict: Dict representing the graph, om the same format as with FactKG.
        """
        subgraph = {}
        for entity in entity_list:
            subgraph[entity] = []
            node = self.kg.get(entity)
            if node is None:
                logger.warn(f"Argument {entity} was in `entity_list`, but not in knowledge graph. ")
                continue
            for relation, neighbours in self.kg[entity].items():
                # `neighbours` is a list of nodes `entity` point to with a `relation`-edge.
                subgraph[entity].append(relation)
        return subgraph


def find_subgraphs(kg, df, method="direct", fill_if_empty=False):
    """
    Find all subgraphs of each datapoint row in `df`.
    This constructs the subgraphs of depth `depth` for each entity in the feature `Entity_set`, for each
    row in `df`.

    Args:
        kg (KnowledgeGraph): Knowledge Graph object with desired knowledge grahp.
        df (pd.DataFrame): Dataframe with the datapoints we want to find subgraphs of.
        use_direct (str): Either `direct` (uses direct subgraph) or `one_hop` (uses one-hop subgraph).
        fill_if_empty (bool): If True and subgraph has no relations, will use the full entity nodes
            as the subgraph.

    Returns:
        dict: Dict of the idx pointing to the subgraphs (which are also dicts)
    """
    logger.info("Begin finding subgraphs. ")
    subgraphs = {}
    n_datapoints = len(df)
    for idx in range(n_datapoints):
        logger.debug(f"On idx {idx + 1}/{n_datapoints}")
        if ((idx + 1) % 100) == 0:
            logger.info(f"On idx {idx + 1}/{n_datapoints}")
        entities = ast.literal_eval(df["Entity_set"][idx])
        if method == "direct":
            subgraph = kg.get_direct_subgraph(entities, fill_if_empty=fill_if_empty)
        elif method == "one_hop":
            subgraph = kg.get_one_hop_subgraph(entities)
        subgraphs[idx] = subgraph

    logger.info("Done finding subgraphs")
    return subgraphs


def save_subgraph_to_csv(subgraphs, filepath):
    """
    Saves subgraphs to file.

    Args:
        subgraphs (dict): Dict of idx pointing to subgraphs (found with `find_subgraphs()`)
        filepath (str): Name of file to be saved.
    """
    logger.info("Saving subgraphs to CSV...")
    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["datapoint_id", "subgraph"])

        for idx, subgraph in subgraphs.items():
            writer.writerow([idx, subgraph])
    logger.info("Done saving subgraphs to CSV. ")


if __name__ == "__main__":
    set_global_log_level("info")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to make subgraph for. ")
    parser.add_argument("--method", choices=["direct", "one_hop"], default="direct",
                        help="Use `direct` subgraph (only to other entities) or `one-hop` (to any other node). ")
    parser.add_argument("--fill_if_empty", type=bool, default=True,
                        help="If using `direct` subgraph method, this will use `one-hop` method if subgraph is empty. ")

    args = parser.parse_args()

    if args.dataset_type == "train":
        filenames = [TRAIN_FILENAME]
    elif args.dataset_type == "val":
        filenames = [VAL_FILENAME]
    elif args.dataset_type == "test":
        filenames = [TEST_FILENAME]
    elif args.dataset_type == "all":
        filenames = [TRAIN_FILENAME, VAL_FILENAME, TEST_FILENAME]

    kg_path = Path(DATA_PATH) / DPBEDIA_FOLDER / DBPEDIA_LIGHT_FILENAME
    logger.info("Loading knowledge graph...")
    kg = pickle.load(open(kg_path, "rb"))
    logger.info("Done loading knowledge graph. ")
    kg_instance = KnowledgeGraph(kg)

    save_folder = Path(SAVE_DATAFOLDER) / SUBGRAPH_FOLDER
    os.makedirs(save_folder, exist_ok=True)
    for filename in filenames:
        df_path = Path(DATA_PATH) / FULL_FOLDER / filename
        df = pd.read_csv(df_path)

        if args.method == "direct":
            if args.fill_if_empty:
                method_name = "direct_filled"
            else:
                method_name = "direct"
        else:
            method_name = "one_hop"
        save_filename = "subgraphs_" + method_name + "_" + filename
        save_path = save_folder / save_filename

        subgraphs = find_subgraphs(kg_instance, df, method=args.method, fill_if_empty=args.fill_if_empty)
        save_subgraph_to_csv(subgraphs, save_path)

    if LOCAL:
        from IPython import embed
        embed()
