import os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from typing import Callable


class SubstituteDataset(Dataset):
    """
    Dataset for the images in 'NIPS 2017: Non-targeted Adversarial Attack competition'.
    """

    def __init__(
        self, root_dir: str, get_predictions: Callable, normalize: bool = True
    ):
        """
        Parameters
        ----------
        root_dir: str.
            Directory with the images of the current substitute epoch.

        get_predictions: Callable.
            Function to get the labels from the the oracle, that is, Oracle(x) = label.

        normalize: bool, default=True.
            If True apply transforms.Normalize with the mean and std from MNIST.
        """

        super(SubstituteDataset, self).__init__()

        self.root_dir = root_dir
        self.get_predictions = get_predictions
        self.images = [file for file in os.listdir(root_dir)]

        if normalize:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx}.png")
        img = Image.open(img_name).convert("L")
        img = self.transform(img)

        if len(img.shape) < 4:
            label = self.get_predictions(img.unsqueeze(dim=0))
        else:
            label = self.get_predictions(img)

        return img, label
