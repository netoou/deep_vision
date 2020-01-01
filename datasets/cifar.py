from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np

import os

CIFAR_MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

CIFAR_STD = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

class Cifar100Dataset(Dataset):
    """CIFAR100 Dataset."""
    def __init__(self, root_dir, set_type='train', transform=None, val_size=0):
        self.root_dir = root_dir
        self.set_type = set_type
        self.transform = transform
        self.val_size = val_size

        self.set_dir = os.path.join(root_dir, 'test' if self.set_type == 'test' else 'train')
        self.data_dict = self._unpickle(self.set_dir)
        self.data = self.data_dict[b'data'][-self.val_size:] if self.set_type == 'val' else self.data_dict[b'data']
        self.label = self.data_dict[b'fine_labels'][-self.val_size:] if self.set_type == 'val' else self.data_dict[b'fine_labels']
        assert len(self.label) == len(self.data), 'Quantity of the data and label does not match!'

        if not transform == None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        # separate normalization to apply dynamic augment policy
        self.normalize = transforms.Normalize(CIFAR_MEAN['cifar100'], CIFAR_STD['cifar100'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Image
        img = self.data[idx].reshape((3,32,32)).transpose(1,2,0)
        img = self.transform(img)

        # Label
        label = torch.tensor(self.label[idx], dtype=torch.long)
        return self.normalize(img), label

    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def set_transform(self, transform):
        self.transform = transform


if __name__ == '__main__':
    dset = Cifar100Dataset('/home/ailab/data/cifar-100-python/')
    dloader = DataLoader(dset,batch_size=4)
    print(len(dset))
    print(len(dloader))
    dloader.dataset.set_transform(1)