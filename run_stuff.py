import torch
import torch.nn as nn
import transformers
from datasets import get_dataloader
from models import get_bert_model
from train import train_simple
from utils import get_logger, set_global_log_level, seed_everything
import argparse
logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--subgraph_type", type=str, default="None")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(57)
    args = get_args()

    if args.subgraph_type.lower() in ["", "none"]:
        args.subgraph_type = None
    train_loader = get_dataloader("train", args.subgraph_type, batch_size=args.batch_size)
    val_loader = get_dataloader("train", args.subgraph_type, batch_size=args.batch_size)
    model_name = "bert_" + str(args.subgraph_type)
    model = get_bert_model(model_name, freeze=args.freeze)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=len(train_loader) * args.n_epochs
    )
    criterion = nn.CrossEntropyLoss()

    train_simple(model, criterion, optimizer, train_loader, val_loader, n_epochs=args.n_epochs, scheduler=lr_scheduler)
    # from IPython import embed
    # embed()
