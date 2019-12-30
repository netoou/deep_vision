import os
import argparse

import torch

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from models.classification.EfficientNet import EfficientNet
from models.classification.MobileNetV3 import mobilenet_v3
from models.classification.ResNeXt import resnext
from datasets.cifar import Cifar100Dataset

from sklearn.metrics import average_precision_score, f1_score, accuracy_score

from datetime import datetime
from tqdm import trange, tqdm
import time


def topk_accuracy(logits: torch.tensor, targets: torch.tensor, k: int):
    _, topk = logits.softmax(dim=1).topk(k=k, dim=1)
    true_cnt = 0
    for pred, tgt in zip(topk, targets):
        if tgt in pred:
            true_cnt += 1

    return true_cnt / len(targets)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='resnext-mini-16-4d')#'mobilenet-v3-small')#'efficientnet-b0')
    parser.add_argument('--cuda', type=bool, default=False)

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=400)

    return parser.parse_args()


def train_epoch(model, epoch, optimizer, criterion, data_loaders, device, log):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        top1 = list()
        logit_list = list()
        gt_targets = list()
        loss_list = list()
        phase_time = time.time()

        t = tqdm(data_loaders[phase], desc=f'Epoch{epoch}, {phase}', position=0)
        for step, (imgs, targets) in enumerate(t):
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)

            loss = criterion(logits, targets)
            if phase == 'train':
                # opt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top1_acc = topk_accuracy(logits, targets, k=1)
            top5_acc = topk_accuracy(logits, targets, k=5)

            logit_list.append(logits.data.cpu())
            top1.append(logits.softmax(dim=1).argmax(dim=1).data.cpu())
            gt_targets.append(targets.data.cpu())
            loss_list.append(loss.cpu().item())

            # print average loss and acc
            t.set_description(
                desc=f"epoch: {epoch}, phase: {phase}, top1: {top1_acc:.4f}, top5: {top5_acc:.4f}, loss: {loss.cpu().item():.4f}")
            #             log.write(f"epoch: {epoch}, phase: {phase}, top1: {top1_acc:.4f}, top5: {top5_acc:.4f}, loss: {loss:.4f}\n")
            log.flush()

        logits = torch.cat(logit_list)
        gt_targets = torch.cat(gt_targets)
        loss = torch.tensor(loss_list).mean()

        top1_acc = topk_accuracy(logits, gt_targets, k=1)
        top5_acc = topk_accuracy(logits, gt_targets, k=5)

        t.set_description(
            desc=f"epoch: {epoch}, phase: {phase}, avg top1: {top1_acc:.4f}, avg top5: {top5_acc:.4f}, avg loss: {loss.item():.4f}")
        log.write(
            f"epoch: {epoch}, phase: {phase}, avg top1: {top1_acc:.4f}, avg top5: {top5_acc:.4f}, avg loss: {loss.item():.4f}\n")

    return top1_acc, top5_acc, loss


mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


if __name__=='__main__':
    arg = parse_args()

    nowdate = datetime.now()
    experiment_time = "date{}{}{}".format(nowdate.year, nowdate.month, nowdate.day)

    save_dir = f'./saved/{experiment_time}/'
    logfile = f'./logs/{arg.model}_trained_on_{experiment_time}.txt'

    device = 'cuda' if arg.cuda else 'cpu'
    dset = 'cifar' if arg.dataset in ['cifar100', 'cifar10'] else None

    if not os.path.isdir('./saved/'):
        os.mkdir('./saved/')
    if not os.path.isdir('./logs/'):
        os.mkdir('./logs/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if arg.model == 'efficientnet-b0':
        model = EfficientNet(100, 1.0, 1.0, dset).to(device)
    elif arg.model == 'mobilenet-v3-small':
        model = mobilenet_v3(100, 'small').to(device)
    elif arg.model == 'mobilenet-v3-large':
        model = mobilenet_v3(100, 'large').to(device)
    elif arg.model == 'resnext-mini-16-4d':
        model = resnext(100, 'resnext-mini-16-4d').to(device)
    elif arg.model == 'resnext50-32-4d':
        model = resnext(100, 'resnext50-32-4d').to(device)

    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=arg.momentum,
                          weight_decay=arg.weight_decay, nesterov=True)

    criterion = nn.CrossEntropyLoss().to(device)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[arg.dataset], std[arg.dataset])
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean[arg.dataset], std[arg.dataset])
    ])

    cifar_train = Cifar100Dataset('/home/ailab/data/cifar-100-python/', transform=transform, set_type='train')
    cifar_val = Cifar100Dataset('/home/ailab/data/cifar-100-python/', transform=transform_val, set_type='test')

    dataloaders = {
        'train': DataLoader(cifar_train, batch_size=arg.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(cifar_val, batch_size=arg.batch_size, num_workers=4),
    }

    with open(logfile, 'w+') as log:
        log.write("experiment starts!!!\n")
        log.write(f"train size : {len(cifar_train)} val size : {len(cifar_val)}\n")
        epoch_infos = dict()
        best_acc = 0.0

        for epoch in range(1, arg.epochs + 1):
            top1_acc, top5_acc, loss = train_epoch(model, epoch, optimizer, criterion, dataloaders, device, log)
            lr_scheduler.step()

            epoch_infos[epoch] = {'top1': top1_acc, 'top5': top5_acc, 'loss': loss}
            if best_acc < top1_acc:
                best_acc = top1_acc
                print("Best model found!!")
                log.write("Best model found!!!!\n")

                save_path = os.path.join(save_dir, f'{arg.model}_epoch{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_infos': epoch_infos,
                    'best_acc': best_acc,
                }, save_path)

                print("model saved at : {}".format(save_path))
                log.write("model saved at : {}\n".format(save_path))



