from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import xmltodict

from PIL import Image
from collections import OrderedDict

from misc.utils import yxyx_to_yxhw

import numpy as np
import cv2

import os
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tv/monitor', ]

#detection
class VOCDataset(Dataset):
    def __init__(self, root:str, dataset='voc2012', input_size=(224,224), transform=None):
        super(VOCDataset, self).__init__()

        self.root = root
        self.dataset = dataset.upper()
        self.input_size = input_size

        self.data_dir = os.path.join(self.root, self.dataset)
        self.ann_dir = os.path.join(self.data_dir, 'Annotations')
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.xmlann_list = [os.path.join(self.ann_dir, i) for i in os.listdir(self.ann_dir)]

        if not transform == None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=input_size),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        """
        Cautions: we need to fix ann matrix sizes for pytorch batch sampling, if the matirx size is dynamic, the batch tensor will break
        :param index: data idx
        :return: image tensor, annotation matrix
        """
        # read anns
        xmlpath = self.xmlann_list[index]
        xml_dict = self._parse_voc_xml(xmlpath)
        filename, _, img_size, objects = self._parse_voc_dict(xml_dict)
        # read image
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        # label
        objects[:,1:] = np.vstack([yxyx_to_yxhw(i) for i in objects[:,1:]])
        objects = torch.tensor(objects, dtype=torch.float)

        return img, objects

    def __len__(self):
        return len(self.xmlann_list)


    def _parse_voc_xml(self, xmlfile:str):
        """
        Function to parse xml file to dict, using xmltodict library
        :param xmlfile:
        :return: python dictionary of xml
        """
        with open(xmlfile) as f:
            xml_dict = xmltodict.parse(f.read())

        return xml_dict

    def _parse_voc_dict(self, ann_dict:dict):
        """
        Transform annotation dictionary to numpy array style annotation blocks,

        :param ann_dict: Annotation dictionary from xml file, recommend to use xmltodict library
        :return: 5 x 20 numpy array, first column is class label, other 1 to 4 columns are bounding box information with following order : ymin, ymax, xmin, xmax
        """
        ann_dict = ann_dict['annotation']

        filename = ann_dict['filename']
        folder = ann_dict['folder']

        img_size = ann_dict['size']
        img_size = (int(img_size['depth']), int(img_size['height']), int(img_size['width']),)

        object_dict = ann_dict['object']
        if type(object_dict) == OrderedDict:
            object_dict = [object_dict]
        # row with -1 indicates end of object
        objects = np.zeros((20, 5), dtype=np.float) - 1
        # order : class, xmax, xmin, ymax, ymin
        for idx, obj in enumerate(object_dict):
            objects[idx][0] = classes.index(obj['name'])
            # VOC format follows Matlab, index starts from 0
            objects[idx][1] = int(obj['bndbox']['ymin']) - 1
            objects[idx][2] = int(obj['bndbox']['xmin']) - 1
            objects[idx][3] = int(obj['bndbox']['ymax']) - 1
            objects[idx][4] = int(obj['bndbox']['xmax']) - 1

        # reduce the length, height of bbox to match transformed image size
        objects[:,1] = objects[:,1] * (self.input_size[0] / img_size[1])
        objects[:, 2] = objects[:, 2] * (self.input_size[1] / img_size[2])
        objects[:, 3] = objects[:, 3] * (self.input_size[0] / img_size[1])
        objects[:, 4] = objects[:, 4] * (self.input_size[1] / img_size[2])

        return filename, folder, img_size, objects


# TODO Complete voc dataset manager, for classification, segmentation, detection
if __name__ == '__main__':
    root = '/home/ailab/data/'

    testvoc = VOCDataset(root)
    testloader = DataLoader(testvoc, batch_size=4, num_workers=4)

    diter = iter(testloader)

    imgs,labels = next(diter)

    print(imgs.shape)
    print(labels.shape)

    # print(xml_dict)
    # print('-'*40)
    # print(xml_dict['annotation']['filename'])
    # print('-' * 40)
    # for i in xml_dict['annotation']['object']:
    #     print(i['bndbox'])
