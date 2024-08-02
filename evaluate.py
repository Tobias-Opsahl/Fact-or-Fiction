import torch
from sklearn.metrics import precision_recall_fscore_support

from datasets import get_df


def evaluate_on_test_set(qa_gnn, model, test_loader, criterion=None, device=None):
    """
    Evaluates a model. Calculates accuracy, precision, recall and F1 score on a test-set.

    Args:
        qa_gnn (bool): True if model is a qa_gnn.
        model (pytorch.model): PyTorch model.
        test_loader (dataloader): Dataloader of test set.
        criterion (callable, optional): If `qa_gnn`, the correct criterion (loss function) must be provided.
        device (str, optional): Device to train on. Defaults to None.

    Returns:
        dict: Dict of metrics
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_df = get_df("test")
    claim_types = ["existence", "substitution", "multi hop", "multi claim", "negation"]
    count_dict = {"existence": 0, "substitution": 0, "multi hop": 0, "multi claim": 0, "negation": 0, "single hop": 0}
    metrics_dict = {ct: {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0} for ct in claim_types + ["single hop"]}
    count_dict = {ct: 0 for ct in claim_types + ["single hop"]}
    total_loss = 0
    total_correct = 0
    offset = 0

    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs in test_loader:
            if qa_gnn:
                inputs, data_graph, labels = inputs
                batch = inputs.to(device)
                data_graph = data_graph.to(device)
                labels = labels.to(device)

                outputs = model(batch, data_graph)
                loss = criterion(outputs, labels)

                probabilities = torch.sigmoid(outputs)
                preds = (probabilities > 0.5).int()
            else:  # Finetuned BERT
                batch = inputs.to(device)

                outputs = model(**batch)
                loss = outputs.loss

                probabilities = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probabilities, dim=1)
                labels = batch["labels"]

            correct_list = (preds == labels)
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_correct += correct_list.sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(labels.shape[0]):
                metadata = test_df["types"].iloc[offset + i]
                for claim_type in claim_types:
                    if claim_type in metadata:
                        metrics_dict[claim_type]["accuracy"] += correct_list[i].item()
                        count_dict[claim_type] += 1
                if "multi hop" not in metadata:
                    metrics_dict["single hop"]["accuracy"] += correct_list[i].item()
                    count_dict["single hop"] += 1
            offset += i + 1

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    # Calculate metrics for each claim type
    for claim_type in metrics_dict:
        if count_dict[claim_type] != 0:
            metrics_dict[claim_type]["accuracy"] /= count_dict[claim_type]

            if claim_type == "single hop":
                type_preds = [pred for pred, meta in zip(all_preds, test_df["types"]) if "multi hop" not in meta]
                type_labels = [label for label, meta in zip(all_labels, test_df["types"]) if "multi hop" not in meta]
            else:
                type_preds = [pred for pred, meta in zip(all_preds, test_df["types"]) if claim_type in meta]
                type_labels = [label for label, meta in zip(all_labels, test_df["types"]) if claim_type in meta]

            if len(type_preds) > 0:
                p, r, f, _ = precision_recall_fscore_support(type_labels, type_preds, average="binary")
                metrics_dict[claim_type]["precision"] = p
                metrics_dict[claim_type]["recall"] = r
                metrics_dict[claim_type]["f1"] = f

    # Overall, all claim types
    metrics_dict["overall"] = {
        "accuracy": total_correct / len(test_loader.dataset),
        "loss": total_loss / len(test_loader.dataset),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return metrics_dict
