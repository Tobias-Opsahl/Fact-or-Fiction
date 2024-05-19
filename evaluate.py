import ast

import torch
from datasets import get_df


def evaluate_on_test_set(qa_gnn, model, test_loader, criterion=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if qa_gnn:
        return evaluate_on_test_set_qa_gnn(model, test_loader, criterion, device)
    else:
        return evaluate_on_test_set_simple(model, test_loader, device)


def evaluate_on_test_set_simple(model, test_loader, device):
    test_df = get_df("test")
    claim_types = ["existence", "substitution", "multi hop", "multi claim", "negation"]
    acc_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    count_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    total_loss = 0
    total_correct = 0
    offset = 0

    model.eval()

    with torch.no_grad():
        for inputs in test_loader:
            batch = inputs.to(device)

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item() * batch["input_ids"].size(0)
            probabilities = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            correct_list = (preds == batch["labels"])
            total_correct += correct_list.sum().item()

            for i in range(inputs["labels"].shape[0]):
                metadata = ast.literal_eval(test_df["Metatada"].iloc[offset + i])
                for claim_type in claim_types:
                    if claim_type in metadata:
                        acc_dict[claim_type] += correct_list[i].item()
                        count_dict[claim_type] += 1
                if "multi hop" not in metadata:
                    acc_dict["single hop"] += correct_list[i].item()
                    count_dict["single hop"] += 1
            offset += i + 1

    for claim_type in acc_dict:
        if count_dict[claim_type] != 0:
            acc_dict[claim_type] /= count_dict[claim_type]
    acc_dict["total_test_accuracy"] = total_correct / len(test_loader.dataset)
    acc_dict["average_test_loss"] = total_loss / len(test_loader.dataset)

    return acc_dict


def evaluate_on_test_set_qa_gnn(model, test_loader, criterion, device):
    test_df = get_df("test")
    claim_types = ["existence", "substitution", "multi hop", "multi claim", "negation"]
    acc_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    count_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    total_loss = 0
    total_correct = 0
    offset = 0

    model.eval()

    with torch.no_grad():
        for inputs, data_graph, labels in test_loader:
            batch = inputs.to(device)
            data_graph = data_graph.to(device)
            labels = labels.to(device)

            outputs = model(batch, data_graph)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch["input_ids"].size(0)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).int()
            correct_list = (preds == labels)
            total_correct += correct_list.sum().item()

            for i in range(labels.shape[0]):
                metadata = ast.literal_eval(test_df["Metatada"].iloc[offset + i])
                for claim_type in claim_types:
                    if claim_type in metadata:
                        acc_dict[claim_type] += correct_list[i].item()
                        count_dict[claim_type] += 1
                if "multi hop" not in metadata:
                    acc_dict["single hop"] += correct_list[i].item()
                    count_dict["single hop"] += 1
            offset += i + 1

    for claim_type in acc_dict:
        if count_dict[claim_type] != 0:
            acc_dict[claim_type] /= count_dict[claim_type]
    acc_dict["total_test_accuracy"] = total_correct / len(test_loader.dataset)
    acc_dict["average_test_loss"] = total_loss / len(test_loader.dataset)
    return acc_dict


def evaluate_chatgpt(df):
    claim_types = ["existence", "substitution", "multi hop", "multi claim", "negation"]
    acc_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    count_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    total_correct = 0

    for i in range(len(df)):
        prediction = df["predictions"].iloc[i]
        label = df["labels"].iloc[i]
        correct = int(prediction == label)
        total_correct += correct

        metadata = ast.literal_eval(df["metadata"].iloc[i])
        for claim_type in claim_types:
            if claim_type in metadata:
                acc_dict[claim_type] += correct
                count_dict[claim_type] += 1
        if "multi hop" not in metadata:
            acc_dict["single hop"] += correct
            count_dict["single hop"] += 1

    for claim_type in acc_dict:
        if count_dict[claim_type] != 0:
            acc_dict[claim_type] /= count_dict[claim_type]
    acc_dict["total_test_accuracy"] = total_correct / len(df)
    print(acc_dict)
    return acc_dict
