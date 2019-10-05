"""
Mask dataset for image inpainting

paper name: Image Inpainting for Irregular Holes Using Partial Convolutions (NVIDIA)
"""
import torchvision
import torch

from torch.utils.data import Dataset

from PIL import Image

import os
import os.path as osp


class Maskdb(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_size=(512, 512)):
        super(Maskdb, self).__init__()
        self.dataDir = data_dir
        self.filelist = os.listdir(path=data_dir)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=input_size),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        path = self.filelist[index]
        mask = Image.open(osp.join(self.dataDir, path))
        mask = self.transform(mask)

        # the masks are not binary, so we should change it to binary(0,1)
        # and expand it to 3 channels
        mask = mask.apply_(lambda x: 1 if x == 1 else 0)
        mask = mask.expand((3, mask.shape[1], mask.shape[2]))

        return mask

    def __len__(self):
        return len(self.filelist)
