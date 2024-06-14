import argparse
import ast
import csv
import os
import pickle
from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords

from baseline.kg import KG
from constants import (DATA_PATH, DBPEDIA_FOLDER, DBPEDIA_LIGHT_FILENAME, SUBGRAPH_FOLDER, TEST_FILENAME,
                       TRAIN_FILENAME, VAL_FILENAME)
from glocal_settings import SMALL, SMALL_SIZE
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
                        subgraph[entity].append([relation])

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
                subgraph[entity].append([relation])
        return subgraph

    def get_relevant_subgraph(self, entity_list, claim, stop_words, nlp):
        """
        Get both direct subgraph and node-edge pairs were `edge` is a word in `claim`.
        This is done in a lemmatized matter.

        Args:
            entity_list (list of str): List of the entities.
            claim (str): The claim (sentence or question input).
            stop_words (list of str): List of words to not search for in edges.
            nlp (spacy nlp model): NLP model for lemmatization, from spacy.

        Returns:
            dict: Dict representing graphs.
        """
        subgraph = {}
        doc = nlp(claim)
        relevant_words = [token.lemma_ for token in doc if token.text.lower() not in stop_words and token.is_alpha]

        for entity in entity_list:
            subgraph[entity] = []
            node = self.kg.get(entity)
            if node is None:
                logger.warn(f"Argument {entity} was in `entity_list`, but not in knowledge graph. ")
                continue
            for relation, neighbours in self.kg[entity].items():
                # `neighbours` is a list of nodes `entity` point to with a `relation`-edge.
                if relation.lower() in relevant_words:
                    subgraph[entity].append([relation])

                for neighbour in neighbours:  # Direct node relation
                    if neighbour in entity_list:
                        subgraph[entity].append([relation])

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
        List: List of the subgraphs (which are also dicts)
    """
    logger.info("Begin finding subgraphs. ")
    subgraphs = []

    if method == "relevant":  # Load NLP models
        nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

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
        elif method == "relevant":
            claim = df["Sentence"][idx]
            subgraph = kg.get_relevant_subgraph(entities, claim=claim, stop_words=stop_words, nlp=nlp)
        subgraphs.append(subgraph)

    logger.info("Done finding subgraphs")
    return subgraphs


def walk_graphs(subgraph_df, kg_instance):
    """
    Loops over a df with subgraphs and walks the graphs.

    Args:
        subgraph_df (pd.Dataframe): The dataframe with subgraphs.
        kg_instance (KG): The KnowledgeGraph with walk method.

    Returns:
        list: List of the walked graphs.
    """
    walked = []
    logger.info("Begin walking graphs")
    for i in range(len(subgraph_df)):
        subgraph = subgraph_df["subgraph"][i]
        if (i % 100) == 0:
            logger.info(f"On idx {i}/{len(subgraph_df)}")
        walked_graphs = kg_instance.search(list(subgraph.keys()), subgraph)
        walked.append(walked_graphs)
    logger.info("Done walking graphs")
    return walked


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
    parser.add_argument("--method", choices=["direct", "one_hop", "relevant"], default="direct",
                        help="Use `direct` subgraph (only to other entities) or `one-hop` (to any other node). ")
    parser.add_argument("--fill_if_empty", action="store_true",
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

    kg_path = Path(DATA_PATH) / DBPEDIA_FOLDER / DBPEDIA_LIGHT_FILENAME
    logger.info("Loading knowledge graph...")
    kg = pickle.load(open(kg_path, "rb"))
    logger.info("Done loading knowledge graph. ")
    knowledge_graph_instance = KnowledgeGraph(kg)
    kg_instance = KG(kg)

    save_folder = Path(DATA_PATH) / SUBGRAPH_FOLDER
    os.makedirs(save_folder, exist_ok=True)
    for filename in filenames:
        df_path = Path(DATA_PATH) / filename
        df = pd.read_csv(df_path)
        if SMALL:
            df = df[:SMALL_SIZE]

        if args.method == "direct":
            if args.fill_if_empty:
                method_name = "direct_filled"
            else:
                method_name = "direct"
        elif args.method == "one_hop":
            method_name = "one_hop"
        elif args.method == "relevant":
            method_name = "relevant"

        save_filename = "subgraphs_" + method_name + "_" + filename
        save_filename = save_filename.replace(".csv", ".pkl")
        if SMALL:
            save_filename = "small_" + save_filename
        save_path = save_folder / save_filename

        subgraphs = find_subgraphs(knowledge_graph_instance, df, method=args.method, fill_if_empty=args.fill_if_empty)
        subgraph_df = pd.DataFrame()
        subgraph_df["subgraph"] = subgraphs
        subgraph_df["walked"] = walk_graphs(subgraph_df, kg_instance)
        subgraph_df.to_pickle(save_path)
