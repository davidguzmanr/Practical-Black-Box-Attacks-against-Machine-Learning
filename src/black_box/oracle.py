import os

import torch
from torch import Tensor

from .model import BlackBoxModel

MODEL_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/blackbox.pt"

ORACLE = BlackBoxModel()
ORACLE.load_state_dict(torch.load(MODEL_PATH))
ORACLE.eval()


def get_oracle_prediction(x: Tensor):
    """

    Paramaters
    ----------
    x: Tensor.
        PyTorch tensor representing an MNIST image or batch of images.

    Returns
    -------
    int, label of the predicted image using the oracle.
    """
    return ORACLE(x).softmax(dim=-1).argmax(dim=-1)
