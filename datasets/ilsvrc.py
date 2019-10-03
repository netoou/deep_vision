import os
import os.path as osp

from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from torchvision import transforms

from PIL import Image


class ClassDataset(Dataset):
    def __init__(self, data_dir, class_label, transform):
        super(ClassDataset, self).__init__()
        f_names = os.listdir(data_dir)
        self.file_list = [osp.join(data_dir, f_name) for f_name in f_names]
        self.label_list = [class_label for i in range(len(f_names))]

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = self.transform(img)
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.file_list)


def make_ilsvrc2012(data_dir, cls_map_path, classes='all', input_size=(224, 224), transform=None, dataset_size=None):
    def _load_cls_map(cls_map_path):
        cls_map = dict()
        with open(cls_map_path, 'r') as f:
            for line in f:
                dir_name, long_label, str_label = line.rstrip().split(' ')
                cls_map[dir_name] = {'long': int(long_label),
                                     'str': str_label}
        return cls_map

    cls_map = _load_cls_map(cls_map_path)

    if transform == None:
        transform = transforms.Compose([
            transforms.Resize(size=input_size),
            transforms.ToTensor(),
        ])

    if classes == 'all':
        classes = list(cls_map.keys())

    # TODO it should be useful if I specify a sampling method, e.g.) uniform (class) sampling....

    return ConcatDataset([ClassDataset(osp.join(data_dir, cls), cls_map[cls]['long'], transform) for cls in classes])

