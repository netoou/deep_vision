from torch.nn import Module
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch

from models.utils import topk_accuracy

from models.Autoaugment.augmentations import AugmentPolicyTransform

import math

DEFAULT_OPT_ARGS = {
    'lr': 1e-2,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'nesterov': True
}


class Trainer:
    def __init__(self,
                 model,
                 datasets,
                 device,
                 epochs,
                 batch_size,
                 n_workers,
                 opt_arg=None,
                 criterion=nn.CrossEntropyLoss,
                 optimizer=optim.SGD):

        self.device = device

        self.model = model.to(self.device)

        self.criterion = criterion().to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers

        if not opt_arg:
            opt_arg = dict(DEFAULT_OPT_ARGS)
        opt_arg['params'] = self.model.parameters()
        self.optimizer = optimizer(**opt_arg)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)

        self.datasets = datasets

    def train(self):
        best_top1 = 0.0
        best_top5 = 0.0
        best_loss = math.inf

        dataloaders = {
            'train': DataLoader(self.datasets['train'], self.batch_size, shuffle=True, num_workers=self.n_workers),
            'val': DataLoader(self.datasets['val'], self.batch_size, shuffle=False, num_workers=self.n_workers),
        }

        for epoch in range(self.epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                loss_list = list()
                logit_list = list()
                gt_targets = list()

                for step, (imgs, targets) in enumerate(dataloaders[phase]):
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    logits = self.model(imgs)

                    loss = self.criterion(logits, targets)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        logit_list.append(logits.data.cpu())
                        loss_list.append(loss.cpu().item())
                        gt_targets.append(targets.data.cpu())

                if phase == 'val':
                    logits = torch.cat(logit_list)
                    gt_targets = torch.cat(gt_targets)

                    loss = torch.tensor(loss_list).mean()
                    top1_acc = topk_accuracy(logits, gt_targets, k=1)
                    top5_acc = topk_accuracy(logits, gt_targets, k=5)

                    if top1_acc > best_top1:
                        best_top1 = top1_acc
                    if top5_acc > best_top5:
                        best_top5 = top5_acc
                    if loss < best_loss:
                        best_loss = loss

            self.lr_scheduler.step()

        return best_top1, best_top5, best_loss

    def set_policy(self, policy):
        # policy : containing 5 sub-policies which has two operations with magnitude and probability params
        # set augment policy at transform of dataset
        self.datasets['train'].set_transform(AugmentPolicyTransform(policy))

if __name__ == '__main__':
    from datasets.cifar import Cifar100Dataset
    from models.classification.MobileNetV3 import mobilenet_v3
    from torchvision import transforms
    device = 'cpu'
    model = mobilenet_v3(100, 'small').to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    datasets = {
        'train': Cifar100Dataset('/home/ailab/data/cifar-100-python/', transform=transform, set_type='train'),
        'val': Cifar100Dataset('/home/ailab/data/cifar-100-python/', transform=transform_val, set_type='test')
    }

    trainer = Trainer(model, datasets, device, 50, 32, 4)

    trainer.train()
