"""
Microsoft COCO dataset
"""
import torchvision
import torch

from torch.utils.data import Dataset

import numpy as np

from pycocotools.coco import COCO

from PIL import Image

import os
import cv2


class COCOdb(Dataset):
    categories = ['__background__',
                  'person', 'bicycle', 'car', 'motorcycle',
                  'airplane', 'bus', 'train', 'truck', 'boat',
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                  'bird', 'cat', 'dog', 'horse', 'sheep',
                  'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                  'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                  'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle', 'wine glass', 'cup', 'fork', 'knife',
                  'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant',
                  'bed', 'dining table', 'toilet', 'tv', 'laptop',
                  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                  'oven', 'toaster', 'sink', 'refrigerator', 'book',
                  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', ]

    def __init__(self, data_dir: str, data_type: str, ann_type: str, input_size=(512, 512), transforms=None):
        super(COCOdb, self).__init__()
        assert data_type in ['train2017', 'val2017', 'test2017', 'train2014', 'val2014', 'test2014']
        assert ann_type in ['instances', 'keypoints']

        self.data_dir = data_dir
        self.data_type = data_type
        self.ann_type = ann_type
        self.input_size = input_size

        self.prefix = 'person_keypoints' if self.ann_type == 'keypoints' else 'instances'
        self.ann_file = f'{self.data_dir}/annotations/{self.prefix}_{self.data_type}.json'

        self.coco_gt = COCO(self.ann_file)
        self.img_ids = sorted(self.coco_gt.getImgIds())
        self.coco_anns = self.coco_gt.anns

        if transforms == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=input_size),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform = transforms

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        filename = self.coco_gt.imgs[img_id]['file_name']
        path = os.path.join(self.data_dir, self.data_type, filename)
        img = Image.open(path)
        img = self.transform(img.convert('RGB'))

        labels = torch.tensor(self._multiclass_label(img_id), dtype=torch.long)

        masks = torch.tensor(self._foreground_mask(img_id, self.input_size), dtype=torch.int8)

        return img, labels, masks

    def __len__(self):
        return len(self.img_ids)

    def _foreground_mask(self, img_id, input_size: tuple):
        """
        Concatenates all object masks into one, It is able to used to separate foreground and background

        :param img_id: image id
        :return: foreground masks ()
        """
        img_ann_ids = self.coco_gt.getAnnIds(img_id)
        masks = np.array([self.coco_gt.annToMask(self.coco_anns[i]) for i in img_ann_ids], dtype=np.uint8).sum(axis=0)
        masks.dtype = np.uint8
        return cv2.resize(masks, input_size)

    def _multiclass_label(self, img_id):
        """
        An image can have multiple objects, so this function can transform annotations of the image to multi-class label

        :param img_id: image id
        :return:
        """
        img_ann_ids = self.coco_gt.getAnnIds(img_id)
        uniq_cls = np.unique([self.coco_anns[i]['category_id'] for i in img_ann_ids])
        uniq_cls = np.array([COCOdb.categories.index(self.coco_gt.cats[i]['name']) for i in uniq_cls])

        labels = np.zeros(len(COCOdb.categories), dtype=np.long)
        labels[uniq_cls] = 1

        return labels


if __name__ == '__main__':
    coco_dir = '/home/ailab/hdd1/coco'
    prefix = 'instances'
    data_type = 'val2017'
    import math
    math.gcd()
    math.fmod()
    bbbb = [1,2,3,4,5]
    bbbb.count(1)
