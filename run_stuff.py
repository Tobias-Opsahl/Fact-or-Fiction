import argparse

import torch
import torch.nn as nn
import transformers
from datasets import get_dataloader, get_graph_dataloader
from glocal_settings import SMALL, SMALL_SIZE
from models import QAGNN, get_bert_model
from train import run_epoch_qa_gnn, run_epoch_simple, train
from utils import get_logger, save_history, seed_everything, set_global_log_level

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="The name of the model, will be saved to file. ")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=1, help="The amount of epochs to train for. ")
    parser.add_argument("--freeze", action="store_true", help="Wether or not to freeze basemodel")
    parser.add_argument("--qa_gnn", action="store_true", help="Will use QA-GNN model. If not, will fine tune Bert.")

    help = "The type of subgraph to load. `none`: no subgraphs. `direct`: Only directly connected subgraphs. "
    help += "`direct_filled`: Direct subgraph if they exists, else the one-neigbhour of every entity. "
    help += "`one_hop`: The edges one away from every entity. "
    parser.add_argument("--subgraph_type", choices=["none", "direct", "direct_filled", "one_hop"],
                        type=str, default="none", help=help)
    help = "Which subgraph to use for none-qagnn model. `discovered`: Use the raw entity and edges, not walked in "
    help += "graph. `connected`: Only use the connected graph representation found from walking in the knowledge "
    help += "graph. `walkable`: Use all of the walkable paths as subgraph evidence. "
    parser.add_argument("--subgraph_to_use", choices=["discovered", "connected", "walkable"],
                        type=str, default="discovered")
    help = "Will use both of the connected and walkable graph (all the time) for qa-gnn model. If not, will use "
    help += "connected if they are not empty, else walkable."
    parser.add_argument("--mix_graphs", action="store_true", help=help)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(57)
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.subgraph_type.lower() == "none":
        if args.qa_gnn:
            raise ValueError("Argument `subgraph_type` can not be `none` when using qa-gnn. ")
        args.subgraph_type = None

    if args.qa_gnn:
        train_loader = get_graph_dataloader(
            "train", subgraph_type=args.subgraph_type, batch_size=args.batch_size, mix_graphs=args.mix_graphs)
        val_loader = get_graph_dataloader(
            "val", subgraph_type=args.subgraph_type, batch_size=args.batch_size, mix_graphs=args.mix_graphs)
        test_loader = get_graph_dataloader(
            "test", subgraph_type=args.subgraph_type, batch_size=args.batch_size, mix_graphs=args.mix_graphs,
            shuffle=False, drop_last=False)
        model_name = "qa_gnn_" + ("mixed" if args.mix_graphs else "")
        model = QAGNN(model_name, hidden_dim=32, out_features=16, dropout=0.6)
    else:
        train_loader = get_dataloader(
            "train", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
        val_loader = get_dataloader(
            "val", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
        test_loader = get_dataloader(
            "test", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size, shuffle=False,
            drop_last=False)
        model_name = "bert_" + str(args.subgraph_type)
        model = get_bert_model(model_name, freeze=args.freeze)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=len(train_loader) * args.n_epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    history, models_dict = train(
        model=model, criterion=criterion, optimizer=optimizer, qa_gnn=args.qa_gnn, train_loader=train_loader,
        val_loader=val_loader, n_epochs=args.n_epochs, scheduler=lr_scheduler)

    model.load_state_dict(models_dict["best_model_state_dict"])
    with torch.no_grad():
        if args.qa_gnn:
            test_loss, test_correct = run_epoch_qa_gnn(
                train=False, dataloader=test_loader, optimizer=optimizer,
                model=model, criterion=criterion, device=device)
        else:
            test_loss, test_correct, = run_epoch_simple(
                train=False, dataloader=test_loader, optimizer=optimizer, model=model, device=device)
        test_accuracy = test_correct / len(test_loader.dataset)
    history["test_accuracy"] = test_accuracy
    dataset_size = "full" if not SMALL else SMALL_SIZE
    save_history(args.model_name, history, dataset_size=dataset_size)
    logger.info(f"Model \"{args.model_name}\": Test accuracy: {test_accuracy * 100}%, {test_loss=:.4f}. ")
