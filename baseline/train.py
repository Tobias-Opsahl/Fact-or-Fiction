from kg import KG

from typing import List
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
import pickle
import ast

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def paths_to_str(paths: List[List[str]]):
    path_strings = [",".join(path) for path in paths]
    paths_str = "|".join(path_strings)
    return paths_str


def create_connected_paths(df):
    connected_paths = []
    for id, row in df.iterrows():
        entities = ast.literal_eval(row["Entity_set"])
        rels = ast.literal_eval(row["Evidence"])
        paths_dict = kg.search(entities, rels)
        connected_paths_str = paths_to_str(paths_dict["connected"])
        connected_paths.append(connected_paths_str)
    return connected_paths


class FactkgDataset(torch.utils.data.Dataset):
    def __init__(self, df, args):
        self.inputs = df['Sentence'].tolist()
        if args.with_evidence:
            self.paths = create_connected_paths(df)
            self.inputs = [self.inputs[i] + " " + self.paths[i] for i in range(len(self.inputs))]
        self.labels = df['Label'].astype(int).tolist()

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class CollateFunctor:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        claims_with_evidence, labels = zip(*batch)
        inputs = self.tokenizer(list(claims_with_evidence), return_tensors='pt',
                                padding=True, truncation=True, max_length=self.max_len)
        inputs['labels'] = torch.tensor(labels)
        return inputs


def train_epoch(model, train_loader, optimizer, lr_scheduler, device):
    model.train()
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # forward pass
        loss = model(**batch).loss

        # backward pass
        loss.backward()

        # update weights
        lr_scheduler.step()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    for batch in tqdm(val_loader):
        batch = batch.to(device)
        outputs = model(**batch)
        total_correct += (outputs.logits.argmax(dim=1) == batch['labels']).sum().item()
        total_samples += batch['labels'].shape[0]

    accuracy = total_correct / total_samples
    return accuracy


def seed_everything(seed_value=2024):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument(
        "--dbpedia_path", default="/fp/projects01/ec30/factkg/dbpedia/dbpedia_2015_undirected_light.pickle")
    parser.add_argument("--data_path", default="/fp/projects01/ec30/factkg/simple/")
    parser.add_argument("--with_evidence", type=bool, default=False)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything()

    # Read dbpedia
    if args.with_evidence:
        with open(args.dbpedia_path, 'rb') as f:
            dbpedia = pickle.load(f)
            kg = KG(dbpedia)

    train_df = pd.read_csv(args.data_path + 'train.csv')
    val_df = pd.read_csv(args.data_path + 'val.csv')

    # Load data.
    train_data = FactkgDataset(train_df, args)
    val_data = FactkgDataset(val_df, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=CollateFunctor(tokenizer, 512)
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=CollateFunctor(tokenizer, 512)
    )

    # Load model.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        cache_dir="./cache",
        trust_remote_code=True,
        num_labels=2
    ).to(device)
    if args.freeze:
        for params in model.base_model.parameters():
            params.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=len(train_loader) * args.epochs
    )

    # Train
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, lr_scheduler, device)
        accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: validation accuracy = {accuracy:.2%}\n")
