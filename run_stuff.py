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
    parser.add_argument("--n_epochs", type=int, default=2, help="The amount of epochs to train for. ")
    parser.add_argument("--qa_gnn", action="store_true", help="Will use QA-GNN model. If not, will fine tune Bert.")

    parser.add_argument("--freeze_base_model", action="store_true", help="Freeze bert base model. ")
    parser.add_argument("--freeze_up_to_pooler", action="store_true", help="Freeze bert up to last pooling layer. ")
    parser.add_argument("--bert_dropout", type=float, default=0, help="Dropout rate for bert classification layer. ")
    parser.add_argument("--gnn_dropout", type=float, default=0.3, help="Dropout rate for GNN layers. ")
    parser.add_argument("--n_gnn_layers", type=int, default=2, help="Number of GNN layers in QA-GNN. ")
    parser.add_argument("--gnn_batch_norm", action="store_true", help="Use Batch Norm between GNN layers. ")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256, help="Size if hidden dimension in GNN layers. ")
    parser.add_argument("--gnn_out_features", type=int, default=256, help="Size of GNNs last layers output. ")
    parser.add_argument("--vectorized", action="store_true", help="Vectorize GNN processing. ")

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
        model = QAGNN(
            args.model_name, n_gnn_layers=args.n_gnn_layers, gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_out_features=args.gnn_out_features, gnn_batch_norm=args.gnn_batch_norm,
            freeze_base_model=args.freeze_base_model, freeze_up_to_pooler=args.freeze_up_to_pooler,
            gnn_dropout=args.gnn_dropout, vectorized=args.vectorized)
    else:
        train_loader = get_dataloader(
            "train", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
        val_loader = get_dataloader(
            "val", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
        test_loader = get_dataloader(
            "test", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size, shuffle=False,
            drop_last=False)
        model = get_bert_model(
            args.model_name, include_classifier=True, num_labels=2, freeze_base_model=args.freeze_base_model,
            freeze_up_to_pooler=args.freeze_up_to_pooler, dropout_rate=args.bert_dropout)

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
