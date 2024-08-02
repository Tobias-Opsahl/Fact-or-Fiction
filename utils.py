import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from constants import RESULTS_FOLDER, SAVED_MODEL_FOLDER


def set_global_log_level(level):
    LOG_LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    if isinstance(level, str):
        level = level.strip().lower()
        level = LOG_LEVEL_MAP[level]
    logging.getLogger().setLevel(level)
    logging.StreamHandler().setLevel(level)


def get_logger(name):
    """
    Get a logger with the given name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Only if logger has not been set up before
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def seed_everything(seed=57):
    """
    Set all the seeds.

    Args:
        seed (int, optional): The seed to set to. Defaults to 57.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.mps.manual_seed(seed)


def count_parameters(model):
    """
    Returns parameters (trainable and not) of a model.

    Args:
        model (pytorch model): The model to count parameters for.

    Returns:
        (int, int, int): Total parameters, trainable parameters, non-trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def _check_just_file(filename):
    """
    Checks that `filename` does not contain a folder, for example `plots/plot.png`. Raises ValueError if it does.
    Also checks that `filename` is either `str` or `pathtlib.Path`.

    Args:
        filename (str or pathlib.Path): The filename to check.

    Raises:
        ValueError: If `filename` contains a folder.
        TypeError: If `filename` is not of type `str` or `pathlib.Path`.
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()  # Convert to string
    if not isinstance(filename, str):
        raise TypeError(f"Filename must be of type `str` or `pathlib.Path`. Was {type(filename)}. ")
    if "/" in filename or "\\" in filename:
        message = f"Filename must not be inside a directory, but only be the filename. Was {filename}. "
        raise ValueError(message)


def create_folder(folder_path, exist_ok=True):
    """
    Create a folder.

    Args:
        folder_path (str or pathlib.Path): The folder-path, including the foldername.
        exist_ok (bool, optional): If True, will not raise Exception if folder already exists. Defaults to True.
    """
    os.makedirs(folder_path, exist_ok=exist_ok)


def make_file_path(folder, filename, check_folder_exists=True):
    """
    Merges a path to a folder `folder` with a filename `filename`.
    If `check_folder_exists` is True, will create the folder `folder` if it is not there.
    Argument `filename` can not be inside a folder, it can only be a filename (to ensure that the correct and full
    folder path gets created).

    Args:
        folder (str or pathlib.Path): Path to the folder.
        filename (str or pathlib.Path): Filename of the file. Must not be inside a folder, for example `plots/plot.png`
            is not allowed, `plots/` should be a part of `folder`.
        check_folder_exists (bool): If `True`, will check that `folder` exists, and create it if it does not.

    Returns:
        pathlib.Path: The merged path.
    """
    _check_just_file(filename)  # Check that filename does not have a folder.
    folder_path = Path(folder)
    if check_folder_exists:
        create_folder(folder_path, exist_ok=True)
    file_path = folder_path / filename
    return file_path


def get_model_path(model_name):
    model_name = model_name.strip().lower()
    folder_name = SAVED_MODEL_FOLDER
    filename = f"{model_name}.pth"
    file_path = make_file_path(folder_name, filename, check_folder_exists=True)
    return file_path


def save_model(state_dict, model_name):
    """
    Saves a models state_dict.

    Args:.
        state_dict (dict): The state-dict of the model to save.
        model_name (str): String name of the model to save.
    """
    file_path = get_model_path(model_name)
    torch.save(state_dict, file_path)


def load_state_dict(model_path, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=torch.device(device))
    return state_dict


def save_history(model_name, history):
    """
    Save a history dictionary from training.

    Args:
        model_name (str): The name of the model whos history is being loaded.
        history (dict): The histories to save.
    """
    folder_name = Path(RESULTS_FOLDER)
    filename = model_name + "_history.pkl"
    file_path = make_file_path(folder_name, filename, check_folder_exists=True)
    with open(file_path, "wb") as outfile:
        pickle.dump(history, outfile)


def load_history(model_name):
    """
    Load models history, made by `src/evaluation.py`.

    Args:
        model_name (str): The name of the model whos history is being loaded.

    Returns:
        dict: Dictionary of history
    """
    folder_name = Path(RESULTS_FOLDER)
    filename = model_name + "_history.pkl"
    file_path = make_file_path(folder_name, filename, check_folder_exists=False)
    history = pickle.load(open(file_path, "rb"))
    return history
