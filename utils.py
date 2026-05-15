"""
Utility functions for ImaginE.
"""

import os
import random
import logging

import torch
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str | None = None, level=logging.INFO):
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count encoder vs new module parameters.

    Returns:
        (encoder_params, new_module_params)
    """
    if hasattr(model, "get_encoder_params"):
        encoder_ids = set(id(p) for p in model.get_encoder_params())
        enc_total = sum(p.numel() for p in model.get_encoder_params() if p.requires_grad)
        new_total = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad and id(p) not in encoder_ids
        )
        return enc_total, new_total

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, 0
