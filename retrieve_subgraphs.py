import ast
import csv
import json
import pickle
from pathlib import Path

import argparse
import pandas as pd
from constants import (DATA_PATH, DBPEDIA_LIGHT_FILENAME, DPBEDIA_FOLDER, FULL_FOLDER, SAVE_DATAFOLDER, TRAIN_FILENAME,
                       VAL_FILENAME, TEST_FILENAME)
from glocal_settings import LOCAL, SMALL
from utils import get_logger, set_global_log_level

logger = get_logger(__name__)


class KnowledgeGraph:
    def __init__(self, kg):
        self.kg = kg

    def get_direct_subgraph(self, entity_list, fill_if_empty=False):
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
            for entity in entity_list:
                node = self.kg.get(entity)
                if node is None:
                    continue
                for relation, neighbours in self.kg[entity].items():
                    # `neighbours` is a list of nodes `entity` point to with a `relation`-edge.
                    subgraph[entity].append(relation)
        return subgraph

    def get_single_subgraph(self, entity, depth=1):
        """
        Retrieve subgraph of node `entity` with edge depth `depth`. Depth first search.

        Args:
            entity (str): Node entity we want subgraph from.
            depth (int, optional): The amount of edges we want to move away from `entity`.

        Returns:
            dict: Dict representing the subgraph.
        """
        subgraph = {}
        queue = [(entity, 0)]  # queue of (current entity, current depth)
        while queue:
            current_entity, current_depth = queue.pop(0)
            if current_depth < depth:
                if not (current_entity in self.kg):
                    continue
                subgraph[current_entity] = self.kg[current_entity]
                for relation, connected_entities in self.kg[current_entity].items():
                    for next_entity in connected_entities:
                        if next_entity not in subgraph:
                            queue.append((next_entity, current_depth + 1))
        return subgraph

    def get_k_hop_subgraph(self, entities, depth=1):
        """
        Finds the subgraphs of multiple entities and combines them to one graph.

        Args:
            entities (list of str): List of the name of the entities to find subgraphs for and combine.
            depth (int, optional): Depths to make in subgraph. Defaults to 2.

        Returns:
            dict: Dict representing the subgraph.
        """
        combined_subgraph = {}
        for entity in entities:
            subgraph = self.get_subgraph(entity, depth)
            for head, relations in subgraph.items():
                if head not in combined_subgraph:
                    combined_subgraph[head] = {}
                for rel, tails in relations.items():
                    if rel not in combined_subgraph[head]:
                        combined_subgraph[head][rel] = []
                    combined_subgraph[head][rel].append(tails)
        return combined_subgraph


def find_subgraphs(kg, df, fill_if_empty=False):
    """
    Find all subgraphs of each datapoint row in `df`.
    This constructs the subgraphs of depth `depth` for each entity in the feature `Entity_set`, for each
    row in `df`.

    Args:
        kg (KnowledgeGraph): Knowledge Graph object with desired knowledge grahp.
        df (pd.DataFrame): Dataframe with the datapoints we want to find subgraphs of.
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
        subgraph = kg.get_direct_subgraph(entities, fill_if_empty=fill_if_empty)
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
    set_global_log_level("debug")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["train", "val", "test"], default="val")
    parser.add_argument("--depth", type=int, default=1)

    args = parser.parse_args()

    if args.dataset_type == "train":
        filename = TRAIN_FILENAME
    elif args.dataset_type == "val":
        filename = VAL_FILENAME
    elif args.dataset_type == "test":
        filename = TEST_FILENAME

    df_path = Path(DATA_PATH) / FULL_FOLDER / filename
    full_df = pd.read_csv(df_path)

    if SMALL:
        df = full_df.iloc[:10]
        save_filename = Path(SAVE_DATAFOLDER) / "val_small_1.csv"
    else:
        df = full_df
        save_filename = Path(SAVE_DATAFOLDER) / "full_val_1.csv"

    kg_path = Path(DATA_PATH) / DPBEDIA_FOLDER / DBPEDIA_LIGHT_FILENAME
    logger.info("Loading knowledge graph...")
    kg = pickle.load(open(kg_path, "rb"))
    logger.info("Done loading knowledge graph. ")
    kg_instance = KnowledgeGraph(kg)

    subgraphs = find_subgraphs(kg_instance, df, fill_if_empty=True)
    save_subgraph_to_csv(subgraphs, save_filename)

    if LOCAL:
        from IPython import embed
        embed()
