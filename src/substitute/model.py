"""
BlackBoxModel architecture for MNIST dataset. It is kinda like AlexNet 
but modified for MNIST. This will be used as an oracle for the substitute model.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from tqdm.notebook import tqdm, trange
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SubstituteModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(SubstituteModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 24 * 24, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes),
        )

        self.add_optimizer()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        # NOTE: Do I have to put the softmax here? In the paper they do that (or it seems like that),
        # in that case I have to change the loss but I am not sure which way to go
        return out

    def add_optimizer(self):
        """
        Sets up the optimizer.

        Creates an instance of the Adam optimizer and sets it as an attribute
        for this class.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_loss(
        self, prediction_batch: torch.Tensor, class_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the value of the loss function

        In this case we are using cross entropy loss. The loss will be averaged
        over all examples in the current minibatch. Use F.cross_entropy to
        compute the loss.
        Note that we are not applying softmax to prediction_batch, since
        F.cross_entropy handles that in a more efficient way. Excluding the
        softmax in predictions won't change the expected transition. (Convince
        yourself of this.)

        Args:
            prediction_batch:
                A torch.Tensor of shape (batch_size, n_classes) and dtype float
                containing the logits of the neural network, i.e., the output
                predictions of the neural network without the softmax
                activation.
            class_batch:
                A torch.Tensor of shape (batch_size,) and dtype int64
                containing the ground truth class labels.
        Returns:
            loss: A scalar of dtype float
        """
        loss = F.cross_entropy(prediction_batch, class_batch.squeeze())

        return loss

    def _fit_batch(self, images_batch, class_batch):
        images_batch, class_batch = images_batch.to(device), class_batch.to(device) 
        self.optimizer.zero_grad()
        pred_batch = self(images_batch)
        loss = self.get_loss(pred_batch, class_batch)
        loss.backward()
        self.optimizer.step()

        return loss

    def train_epoch(
        self, train_data: DataLoader, epoch: int, batch_size: Optional[int] = None
    ) -> float:
        """
        Fit on training data for an epoch.

        Parameters
        ----------
        train_data: DataLoader
            DataLoader that contains the data for the current substitute epoch.

        epoch: int
            Epoch
        """
        self.train()
        desc = f"Epoch {epoch}"
        total = len(train_data) * batch_size if batch_size else len(train_data)
        bar_fmt = "{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]"

        trn_loss = 0
        trn_done = 0

        pbar = tqdm(
            train_data,
            desc=desc,
            total=total,
            leave=False,
            miniters=1,
            unit="ex",
            unit_scale=True,
            bar_format=bar_fmt,
            position=1,
        )

        for (images, labels) in pbar:
            loss = self._fit_batch(images, labels)
            trn_loss += loss.item() * images.shape[0]
            trn_done += images.shape[0]

            pbar.set_postfix({"loss": "%.3g" % (trn_loss / trn_done)})

        return trn_loss / trn_done

    def train_model(
        self, train_data: DataLoader, epochs: int, batch_size: Optional[int] = None
    ) -> float:
        """
        Fit on training data for an epoch.

        Parameters
        ----------
        train_data: DataLoader
            DataLoader that contains the data for the current substitute epoch.

        epoch: int
            Epoch
        """
        trnbar_fmt = "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        for epoch in tqdm(
            range(epochs),
            desc="Training",
            total=epochs,
            leave=False,
            unit="epoch",
            position=0,
            bar_format=trnbar_fmt,
        ):
            train_loss = self.train_epoch(train_data, epoch)

    def jacobian_dataset_augmentation(
        self, substitute_dataset: Dataset, p: int, lambda_: float, root_dir: str
    ) -> None:
        """
        Jacobian dataset augmentation for 'substitute epoch' p + 1.

        Parameters
        ----------
        substitute_dataset: Dataset
            PyTorch dataset that contains the substitute_dataset for 'substitute epoch' p.

        p: int
            Substitute epoch

        lambda_: float
            Size of the perturbation

        root_dir: str
            Directory where the images will be stored.
        """
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        for i in trange(len(substitute_dataset), desc="Jacobian dataset augmentation", leave=False):
            image, label = substitute_dataset.__getitem__(i)
            image, label = image.to(device), label.to(device)
            
            jacobian = torch.autograd.functional.jacobian(self, image.unsqueeze(dim=1)).squeeze()
            new_image = image + lambda_ * torch.sign(jacobian[label])

            # It seems that saving in png loses some information
            save_image(image, fp=f"{root_dir}/{i}.png")
            save_image(new_image, fp=f"{root_dir}/{i + len(substitute_dataset)}.png")
