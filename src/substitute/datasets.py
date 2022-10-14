import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from typing import Callable

# Initial sample for the subtitute dataset, they correspond to 15 images in
# test set of MNIST for each digit 0-9, so a total of 150 images
SAMPLE = {
    0: [3, 10, 13, 25, 28, 55, 69, 71, 101, 126, 136, 148, 157, 183, 188],
    1: [2, 5, 14, 29, 31, 37, 39, 40, 46, 57, 74, 89, 94, 96, 107],
    2: [1, 35, 38, 43, 47, 72, 77, 82, 106, 119, 147, 149, 172, 174, 186],
    3: [18, 30, 32, 44, 51, 63, 68, 76, 87, 90, 93, 112, 142, 158, 173],
    4: [4, 6, 19, 24, 27, 33, 42, 48, 49, 56, 65, 67, 85, 95, 103],
    5: [8, 15, 23, 45, 52, 53, 59, 102, 120, 127, 129, 132, 152, 153, 155],
    6: [11, 21, 22, 50, 54, 66, 81, 88, 91, 98, 100, 123, 130, 131, 138],
    7: [0, 17, 26, 34, 36, 41, 60, 64, 70, 75, 79, 80, 83, 86, 97],
    8: [61, 84, 110, 128, 134, 146, 177, 179, 181, 184, 226, 232, 233, 242, 257],
    9: [7, 9, 12, 16, 20, 58, 62, 73, 78, 92, 99, 104, 105, 108, 113],
}
INDICES = [item for sublist in list(SAMPLE.values()) for item in sublist]


class SubstituteDataset(Dataset):
    """
    Dataset for the images in 'NIPS 2017: Non-targeted Adversarial Attack competition'.
    """

    def __init__(
        self, root_dir: str, get_predictions: Callable, transform: Callable = None
    ):
        """
        Parameters
        ----------
        root_dir: str.
            Directory with the images of the current substitute epoch.

        get_predictions: Callable.
            Function to get the labels from the the oracle, that is, Oracle(x) = label.

        transform: Callable, default=None.
            Optional transformation to apply.
        """

        super(SubstituteDataset, self).__init__()

        self.root_dir = root_dir
        self.get_predictions = get_predictions
        self.transform = transform
        self.images = [file for file in os.listdir(root_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx}.pt")
        img = torch.load(img_name)

        if self.transform:
            img = self.transform(img)

        img_batch = img.unsqueeze(dim=0)
        # Get the prediction with the oracle
        label = self.get_predictions(img_batch)

        return img, label
