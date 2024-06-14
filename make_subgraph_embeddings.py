import argparse
import os
import pickle
from pathlib import Path

import torch
from constants import EMBEDDINGS_FILENAME, DATA_PATH
from datasets import get_precomputed_embeddings, get_subgraphs, calculate_embeddings
from models import get_bert_model
from transformers import AutoTokenizer
from utils import get_logger, seed_everything, set_global_log_level

logger = get_logger(__name__)


def get_entity_and_relations_from_walked_graph(graph):
    # Graph should be walked, and contain [node1, relation, node2] entries
    nodes = []
    relations = []
    for entry in graph:
        node1, relation, node2 = entry
        if node1 not in nodes:
            nodes.append(node1)
        if node2 not in nodes:
            nodes.append(node2)
        if relation not in relations:
            relations.append(relation)
    return nodes, relations


def get_all_embeddings(subgraph_df, tokenizer, model, batch_size=32):
    file_path = Path(DATA_PATH) / EMBEDDINGS_FILENAME
    if os.path.exists(file_path):
        embedding_dict = get_precomputed_embeddings()
    else:
        embedding_dict = {}

    all_entities_and_relations = set()  # Get all text (entities and relations) from all graphs
    for graph in subgraph_df["walked"]:
        graph_data = graph["connected"] + graph["walkable"]
        entities, relations = get_entity_and_relations_from_walked_graph(graph_data)
        all_entities_and_relations.update(entities)
        all_entities_and_relations.update(relations)

    # Convert set to list and remove already computed embeddings
    all_text = [item for item in all_entities_and_relations if item not in embedding_dict]

    # Get embeddings in batches.
    n_text_from_df = len(all_entities_and_relations)
    n_text = len(all_text)
    logger.info(f"Begin calculating embeddings for {n_text} new words, out of {n_text_from_df} words found in graphs. ")
    for i in range(0, n_text, batch_size):
        if (i % 1) == 0:
            logger.info(f"On idx {i}/{n_text}")
        batch_texts = all_text[i:i + batch_size]
        embeddings = calculate_embeddings(batch_texts, tokenizer, model)
        for text, embedding in zip(batch_texts, embeddings):
            embedding_dict[text] = embedding.cpu().numpy()  # Move embeddings to CPU and convert to numpy for storage

    with open(file_path, "wb") as outfile:
        pickle.dump(embedding_dict, outfile)

    return embedding_dict


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(57)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to make subgraph for. ")
    parser.add_argument("--subgraph_type", choices=["direct", "direct_filled", "one_hop", "relevant"], default="direct",
                        help="The subgraph retrieval method to load. ")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to calculate embeddings. ")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset_type == "all":
        dataset_types = ["train", "val", "test"]
    else:
        dataset_types = [args.dataset_type]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for dataset_type in dataset_types:
        subgraph_df = get_subgraphs(dataset_type, args.subgraph_type)
        model = get_bert_model("bert").to(device)
        embeddings = get_all_embeddings(subgraph_df, tokenizer, model, batch_size=args.batch_size)
