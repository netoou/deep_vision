from models.detection.mobilenetv2_mini_ssd import mobilenetv2_mini_ssd
from models.detection.ssd import multiboxLoss_batch, ssd_box_matching_batch
from datasets.voc2012 import VOCDataset

import torch
from torch.utils import data
from datetime import datetime
import time

if __name__=='__main__':
    nowdate = datetime.now()
    logfile = './logs/exp{}{}{}{}.txt'.format(nowdate.year, nowdate.month, nowdate.day, nowdate.hour)
    with open(logfile, 'w+') as log:
        print('Start training!!')

        max_epoch = 20
        step_size = 300
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        vocset = VOCDataset('/home/ailab/data/', input_size=(192, 192))

        train_set = data.Subset(vocset, range(15000))
        test_set = data.Subset(vocset, range(15000, 16000))
        val_set = data.Subset(vocset, range(16000, len(vocset)))

        print(f"train set size : {len(train_set)},"
              f" val set size : {len(val_set)},"
              f" test set size : {len(test_set)}")

        model = mobilenetv2_mini_ssd(21, (192, 192)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        dataloaders = {
            'train': data.DataLoader(train_set, batch_size=4),
            'val': data.DataLoader(val_set, batch_size=4)
        }

        for epoch in range(max_epoch):
            for phase in ['train', 'val']:
                epoch_start = time.time()
                phase_loss = 0.0

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for step, (imgs, labels) in enumerate(dataloaders[phase]):
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        fwd_time = time.time()
                        cls_score, reg_coord = model(imgs)
                        fwd_time = time.time() - fwd_time

                        match_time = time.time()
                        matching_box = ssd_box_matching_batch(model.default_anchors, labels)
                        match_time = time.time() - match_time

                        loss_time = time.time()
                        loss = multiboxLoss_batch(matching_box, cls_score, reg_coord, labels)
                        loss_time = time.time() - loss_time

                        if phase == 'train':
                            backward_time = time.time()
                            loss.backward()
                            optimizer.step()
                            backward_time = time.time() - backward_time
                            if step % 10 == 0:
                                print(f"step : {step}, current loss : {loss:.4f},"
                                      f" timeinfos : [forward : {fwd_time:.4f},"
                                      f" match : {match_time:.4f},"
                                      f" loss : {loss_time:.4f},"
                                      f" backward : {backward_time:.4f}]")

                    phase_loss += loss.cpu().item()

                phase_loss /= len(dataloaders[phase].dataset)
                epoch_time = time.time() - epoch_start

                print(f"[epoch{epoch}][{phase}] loss : {phase_loss:.4f} time : {epoch_time:.4f}")