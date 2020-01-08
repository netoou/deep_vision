from torch.nn import Module
from torch import nn
from torch import optim

import torch
import numpy as np

from models.Autoaugment.MiniNetTrainer import Trainer
from models.Autoaugment.augmentations import RandAugmentPolicyTransform


def gridsearch_magnitude(search_space: dict, model, datasets, model_param):
    N = search_space['N']
    M = search_space['M']

    best_n = 0
    best_m = 0
    best_top1 = 0.0
    # Grid search, better at distributed env
    for n in N:
        for m in M:
            policy = RandAugmentPolicyTransform(n, m)
            # should modify Trainer to magnitude scheduling
            trainer = Trainer(model(**model_param), datasets, 'cpu', 100, 32, 4)
            trainer.set_randaugment(policy)
            top1, top5, loss = trainer.train()

            if top1 > best_top1:
                best_top1 = top1
                best_m = m
                best_n = n

    return best_top1, best_n, best_m #, best_model


SEARCH_SPACE = {
    'N': [i for i in range(1, 5)],
    'M': [i / 10 for i in range(1, 11)],
}


if __name__=='__main__':
    from datasets.cifar import SmallCifar100
    from models.classification.MobileNetV3 import mobilenet_v3
    from torchvision import transforms
    from torch.utils import data

    device = 'cpu'
    model = mobilenet_v3  # (100, 'small').to(device)
    child_param = {
        'n_classes': 100,
        'arc': 'small2',
    }

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    datasets = {
        'train': SmallCifar100('/home/ailab/data/cifar-100-python/', transform=transform, set_type='train'),
        'val': SmallCifar100('/home/ailab/data/cifar-100-python/', transform=transform_val, set_type='test')
    }

    gridsearch_magnitude(SEARCH_SPACE, model, datasets, child_param)


