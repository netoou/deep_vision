import torch
from torch import nn
import torchvision
from torchvision import transforms

from datetime import datetime

from models.MoblieNet import MobileNet, MiniMobileNet
from models.MobileNetV2 import MobileNetV2

from datasets.cifar import Cifar100Dataset

import time

# Hyperparams
max_epoch = 10
batch_size = 64


# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# VOC Dataset Setting
"""
voc_class = ['background','aeroplane','bicycle','bird','boat',
             'bottle','bus','car','cat','chair',
             'cow','diningtable','dog','horse','motorbike',
             'person','pottedplant','sheep','sofa','train','tvmonitor']
voc_class = {cls : idx for idx,cls in enumerate(voc_class)}

VOC2012 = '/home/ailab/data/'
torchvision.datasets.voc.DATASET_YEAR_DICT['2012']['base_dir'] = 'VOC2012'
voc_train = torchvision.datasets.voc.VOCDetection(VOC2012, image_set='train', transform=transforms.Compose([transforms.Resize((224,224)),
                                                                                        transforms.ToTensor(),]))
train_loader = torch.utils.data.DataLoader(voc_train, batch_size=batch_size, shuffle=False, num_workers=4)

voc_val = torchvision.datasets.voc.VOCDetection(VOC2012, image_set='val', transform=transforms.Compose([transforms.Resize((224,224)),
                                                                                        transforms.ToTensor(),]))
val_loader = torch.utils.data.DataLoader(voc_val, batch_size=batch_size, shuffle=False, num_workers=4)
"""
def voc_annTotarget(ann, batch_size, n_classes=21):
    target = torch.zeros([batch_size, n_classes], dtype=torch.float)
    for obj in ann['annotation']['object']:
        for batch_idx,obj_name in enumerate(obj['name']):
            target[batch_idx][voc_class[obj_name]] = 1

    return target

# Model Setting
model = MiniMobileNet(n_classes=100, device=device).to(device)
# model = torchvision.models.resnet18().to(device)
# model.fc = nn.Linear(512, 100)
# model = model.to(device)
#criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

cifar_root = '/home/ailab/data/cifar-100-python/'
cifar_train = Cifar100Dataset(cifar_root, set_type='train')
train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=False, num_workers=8)
cifar_val = Cifar100Dataset(cifar_root, set_type='val')
val_loader = torch.utils.data.DataLoader(cifar_val, batch_size=batch_size, shuffle=False, num_workers=8)

if __name__ == '__main__':
    nowdate = datetime.now()
    logfile = './logs/exp{}{}{}{}.txt'.format(nowdate.year, nowdate.month, nowdate.day, nowdate.hour)
    with open(logfile, 'w+') as log:
        print('Start training!!')
        val_acc_history = []
        dataloaders = {'train' : train_loader, 'val' : val_loader}
        for epoch in range(max_epoch):


            for phase in ['train', 'val']:
                epoch_start = time.time()
                phase_loss = 0.0
                correct = 0

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for imgs, targets in dataloaders[phase]:
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        out = model(imgs)
                        loss = criterion(out, targets)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    _, predictions = torch.max(out, 1)
                    phase_loss += loss.cpu().item()
                    correct += torch.sum(predictions == targets.data)

                phase_loss /= len(dataloaders[phase].dataset)
                correct /= len(dataloaders[phase].dataset)

                epoch_time = time.time() - epoch_start

                print('[epoch{}][{}] loss : {:.6f}, time : {:.4f}'.format(epoch, phase, phase_loss, epoch_time))
                log.write('[epoch{}][{}] loss : {:.6f}, time : {:.4f}'.format(epoch, phase, phase_loss, epoch_time))
                log.flush()




