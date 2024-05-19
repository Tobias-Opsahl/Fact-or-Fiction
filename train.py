
import numpy as np
import torch

from constants import N_EARLY_STOP_DEFAULT
from utils import get_logger, save_model

logger = get_logger(__name__)


def run_epoch_simple(train, dataloader, optimizer, model, device):
    total_loss = 0
    total_correct = 0
    if train:
        model.train()
    else:
        model.eval()

    for inputs in dataloader:
        batch = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch["input_ids"].size(0)
        probabilities = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probabilities, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
    return total_loss, total_correct


def run_epoch_qa_gnn(train, dataloader, optimizer, model, criterion, device):
    total_loss = 0
    total_correct = 0
    if train:
        model.train()
    else:
        model.eval()

    for inputs, data_graph, labels in dataloader:
        batch = inputs.to(device)
        data_graph = data_graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch, data_graph)

        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch["input_ids"].size(0)
        probabilities = torch.sigmoid(outputs)
        preds = (probabilities > 0.5).int()
        total_correct += (preds == labels).sum().item()
    return total_loss, total_correct


def train(model, criterion, optimizer, qa_gnn, train_loader, val_loader=None, n_epochs=10, scheduler=None,
          n_early_stop=None, save_models=True, device=None, non_blocking=False, verbose=1):
    """
    Trains a model and calculate training and valudation stats, given the model, loader, optimizer
    and some hyperparameters.

    Args:
        model (model): The model to train. Freeze layers ahead of calling this function.
        criterion (callable): Pytorch loss function.
        optimizer (optim): Pytorch Optimizer.
        qa_gnn (bool): Wether the model is a QA-GNN model with graphs (True), or a language model (False).
        train_loader (dataloader): Data loader for training set
        val_loader (dataloader, optional): Optinal validation data loader.
            If not None, will calculate validation loss and accuracy after each epoch.
        n_epochs (int, optional): Amount of epochs to run. Defaults to 10.
        scheduler (scheduler, optional): Optional learning rate scheduler.
        n_early_stop (int): The number of consecutive iterations without validation loss improvement that
            stops the training (early stopping). Will only work if `val_loader` is None.
            Set to `False` for deactivating it, and `None` for default value from `constants.py`.
        save_models (bool): If True and `val_loader` is not None, will save the best models state dicts.
        non_blocking (bool): If True, allows for asyncronous transfer between RAM and VRAM.
            This only works together with `pin_memory=True` to dataloader and GPU training.
        verbose (int): If 0, will not log anything. If not 0, will log last epoch with INFO and the others with DEBUG.

    Returns:
        dict: A dictionary of the training history. Will contain lists of training loss and accuracy over
            epochs, and for validation loss and accuracy if `val_loader` is not None.
        dict: A dictionary with trained model, and optional best-model state-dicts if `val_loader` is not None.
            On the form: {"final_model": model, "best_model_accuracy_state_dict": a, "best_model_loss_state_dict": b}
    """
    if n_early_stop is None:
        n_early_stop = N_EARLY_STOP_DEFAULT
    elif not n_early_stop:
        n_early_stop = n_epochs
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=non_blocking)
    train_class_loss_list = []  # Initialize training history variables
    train_class_accuracy_list = []
    val_class_loss_list = []  # These will remain empty if `val_loader` is None
    val_class_accuracy_list = []
    best_epoch_number = -1
    best_val_loss = np.inf
    best_val_accuracy = -1
    best_model = None  # This will only be saved if `val_loader` is not None
    n_stagnation = 0
    if verbose != 0:
        logger.info(f"Starting training with device {device}.")

    for epoch in range(n_epochs):  # Train
        if qa_gnn:
            train_loss, train_correct = run_epoch_qa_gnn(
                train=True, dataloader=train_loader, optimizer=optimizer, model=model,
                criterion=criterion, device=device)
        else:
            train_loss, train_correct = run_epoch_simple(
                train=True, dataloader=train_loader, optimizer=optimizer, model=model, device=device)
        average_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        train_class_loss_list.append(average_train_loss)
        train_class_accuracy_list.append(train_accuracy)

        if val_loader is not None:  # Eval
            with torch.no_grad():
                if qa_gnn:
                    val_loss, val_correct = run_epoch_qa_gnn(
                        train=False, dataloader=val_loader, optimizer=optimizer, model=model,
                        criterion=criterion, device=device)
                else:
                    val_loss, val_correct = run_epoch_simple(
                        train=False, dataloader=val_loader, optimizer=optimizer, model=model, device=device)

                average_val_loss = val_loss / len(val_loader.dataset)
                val_accuracy = 100 * val_correct / len(val_loader.dataset)
                val_class_loss_list.append(average_val_loss)
                val_class_accuracy_list.append(val_accuracy)

                if average_val_loss >= best_val_loss:  # Check for stagnation _before_ updating best_val_loss
                    n_stagnation += 1
                else:  # Better than best loss
                    n_stagnation = 0
                if n_stagnation == n_early_stop:  # Early stopping, abort training
                    if verbose == 0:  # No output
                        break
                    logger.info(f"Epoch [{epoch + 1} / {n_epochs}]\n")
                    logger.info(f"Train loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%")
                    logger.info(f"Validation loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%\n")
                    logger.info(f"Early stopping after {n_stagnation} rounds of no validation loss improvement.\n")
                    break

                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    best_epoch_number = epoch + 1
                    best_val_accuracy = val_accuracy
                    best_model = model.state_dict()

        if scheduler is not None:
            scheduler.step()

        if verbose == 0:  # Do not log
            continue

        message = f"Epoch [{epoch + 1} / {n_epochs}]\n"
        message += f"Train loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%\n"
        if val_loader is not None:
            message += f"Validation loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%"
        message += "\n"

        if epoch + 1 == n_epochs:  # Last epoch
            logger.info(message)
        else:
            logger.debug(message)

    history = {"train_class_loss": train_class_loss_list, "train_class_accuracy": train_class_accuracy_list,
               "val_class_loss": val_class_loss_list, "val_class_accuracy": val_class_accuracy_list,
               "best_epoch": best_epoch_number, "best_val_accuracy": best_val_accuracy,
               "best_val_loss": best_val_loss, "model_name": model.name}

    models_dict = {"final_model": model}
    if val_loader is not None:
        models_dict["best_model_state_dict"] = best_model
        if save_models:
            save_model(best_model, model.name)

    return history, models_dict
