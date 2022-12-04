import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
import torchvision
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2



class AbstractDataset(VisionDataset):
    def __init__(self, transforms_cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractDataset, self).__init__('', transforms=transforms,
                                              transform=transform, target_transform=target_transform)
        # fix for re-production
        np.random.seed(seed)

        self.images = list()
        self.targets = list()
        self.split = None # cfg['split']
        if self.transforms is None:
            self.transforms = torchvision.transforms.Compose(
                [getattr(torchvision.transforms, _['name'])(**_.get('params', {})) for _ in transforms_cfg] 
                # +
                # [ToTensorV2()]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        tgt = self.targets[index]
        return path, tgt

    def load_item(self, items):
        images = list()
        for item in items:
            img = cv2.imread(item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=img)['image']
            images.append(image)
        return torch.stack(images, dim=0)
