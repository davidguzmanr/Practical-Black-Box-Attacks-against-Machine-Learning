from collections import OrderedDict
import os

import numpy as np
import torch
import torchinfo
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

from src.black_box.model import BlackBoxModel
from src.black_box.oracle import get_oracle_prediction

from src.substitute.model import SubstituteModel
from src.substitute.datasets import SubstituteDataset


# STEP 1: Initial Collection 
substitute_dataset = SubstituteDataset(
    root_dir='src/substitute/data/training_set_0',
    get_predictions=get_oracle_prediction
)
train_dataloader = DataLoader(substitute_dataset, batch_size=8)

# STEP 2: Architecture Selection
substitute_model = SubstituteModel()


for p in trange(5, desc='Substitute Training'):
    # STEP 3: Labeling with oracle
    substitute_dataset = SubstituteDataset(
        root_dir=f'src/substitute/data/training_set_{p}',
        get_predictions=get_oracle_prediction
    )
    train_dataloader = DataLoader(substitute_dataset, batch_size=8)
    
    # STEP 4: Training the substitute model
    substitute_model.train_model(train_dataloader, epochs=10)
    
    # STEP 5: Jacobian dataset augmentation
    substitute_model.jacobian_dataset_augmentation(
        substitute_dataset=substitute_dataset,
        p=(p + 1),
        lambda_=0.5,
        root_dir=f'src/substitute/data/training_set_{p+1}'
    )
    
    # STEP 6: ???
    # STEP 7: Profit