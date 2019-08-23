import torch
from torch import nn
import torchvision
from torchvision import transforms

from models.MoblieNet import MobileNet
from models.MobileNetV2 import MobileNetV2

import time

# Hyperparams
max_epoch = 10
batch_size = 4


# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset Setting
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

def voc_annTotarget(ann, batch_size, n_classes=21):
    target = torch.zeros([batch_size, n_classes], dtype=torch.float)
    for obj in ann['annotation']['object']:
        for batch_idx,obj_name in enumerate(obj['name']):
            target[batch_idx][voc_class[obj_name]] = 1

    return target

# Model Setting
model = MobileNet(n_classes=len(voc_class), device=device).to(device)
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.RMSprop(model.parameters())




if __name__ == '__main__':

    val_acc_history = []
    dataloaders = {'train' : train_loader, 'val' : val_loader}

    for epoch in range(max_epoch):


        for phase in ['train', 'val']:
            epoch_start = time.time()
            phase_loss = 0.0
            pred_gap = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for imgs, ann in dataloaders[phase]:
                targets = voc_annTotarget(ann, batch_size).to(device)
                imgs = imgs.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    out = model(imgs)
                    loss = criterion(out, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                predictions = out.data.softmax(dim=1)
                phase_loss += loss.cpu().item()

                pred_gap += nn.ReLU()(targets - predictions).sum().cpu().item()
                print("step working")

            phase_loss /= len(dataloaders[phase].dataset)
            pred_gap /= len(dataloaders[phase].dataset)

            epoch_time = time.time() - epoch_start

            print('[epoch{}][{}] loss : {:.4f}, gap : {:.4f}, time : {:.4f}'.format(epoch, phase, phase_loss, pred_gap, epoch_time))





