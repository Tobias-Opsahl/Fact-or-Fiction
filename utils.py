import logging
import os
import random

import numpy as np
import torch


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
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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
