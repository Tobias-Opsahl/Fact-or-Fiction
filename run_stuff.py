import torch
import torch.nn as nn
import transformers
from datasets import get_dataloader, get_graph_dataloader
from models import get_bert_model, QAGNN
from train import train
from utils import get_logger, set_global_log_level, seed_everything
import argparse
logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--subgraph_type", choices=["none", "direct", "direct_filled", "one_hop"],
                        type=str, default="none")
    parser.add_argument("--subgraph_to_use", choices=["discovered", "connected", "walkable"],
                        type=str, default="discovered")
    parser.add_argument("--qa_gnn", action="store_true")
    parser.add_argument("--mix_graphs", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(57)
    args = get_args()

    if args.subgraph_type.lower() == "none":
        if args.qa_gnn:
            raise ValueError("Argument `subgraph_type` can not be `none` when using qa-gnn. ")
        args.subgraph_type = None

    if args.qa_gnn:
        train_loader = get_graph_dataloader(
            "train", subgraph_type=args.subgraph_type, batch_size=args.batch_size, mix_graphs=args.mix_graphs)
        val_loader = get_graph_dataloader(
            "val", subgraph_type=args.subgraph_type, batch_size=args.batch_size, mix_graphs=args.mix_graphs)
        model_name = "qa_gnn_" + ("mixed" if args.mix_graphs else "")
        model = QAGNN(model_name, hidden_dim=32, out_features=16, dropout=0.6)
    else:
        train_loader = get_dataloader(
            "train", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
        val_loader = get_dataloader(
            "val", args.subgraph_type, subgraph_to_use=args.subgraph_to_use, batch_size=args.batch_size)
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
    # from IPython import embed
    # embed()
