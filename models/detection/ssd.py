"""
Single Shot Multi-box detector
"""
import torch
import torchvision

from torch import nn
from torch.nn import Module
from torch.autograd import Function
from torch.nn.functional import smooth_l1_loss, nll_loss, log_softmax

from misc.utils import iou

import numpy as np

# Base coordinates
# (height, width)
# (y, x)

class SSD(Module):
    def __init__(self, n_classes: int,
                 input_size: tuple,
                 base_network: nn.Module,
                 extra_feature_layers: nn.ModuleList,
                 pred_layers: nn.ModuleList):
        super(SSD, self).__init__()
        """
        
        :param n_classes: The number of classes of your problem 
        :param base_network: The front part of CNN without classification layers, the purpose of this net is extract feature map from input images
        :param feature_layers: List of convolution layers, for every conv layer will generate feature map, this feature maps are used to make classification, bounding box regression result
        :param pred_layers: List of convolution layers, each layer takes the feature map from feature layers, then out the cls, reg result
        """
        assert len(
            input_size) == 2, f"Input : {input_size}, input should be length 2 tuple and should be (height, width) order"

        self.n_classes = n_classes
        self.input_size = input_size
        self.base_network = base_network
        self.extra_feature_layers = extra_feature_layers
        self.pred_layers = pred_layers

        self.featuremap_sizes, self.reduction_ratio = self._set_featuremap_sizes()
        self.default_anchors = self._set_default_anchor_boxes(self.featuremap_sizes, self.reduction_ratio)

        # TODO check the size of featrue maps of between each layers and net, if it does not match, we need to rasie error messgae
        # 일단 기능구현만 하고 에러체크는 나중에 하기

    def forward(self, input):
        # TODO Complete SSD flow
        feature_maps = [self.base_network(input)]

        # save feature maps of each extra layer in the list
        for layer in self.extra_feature_layers:
            feature_maps.append(layer(feature_maps[-1]))

        # last convolution layer will make the prediction tensor of bbox, classification
        for idx, layer in enumerate(self.pred_layers):
            feature_maps[idx] = layer(feature_maps[idx])
        # each feature map will have a shape of (batch_size, 4*(n_classes+4), w, h)
        # values of second dimension are meaning class scores and bbox coordinates
        # from 0 to 4*n_classes indices are for class scores
        # from 4*n_classes to end indices are bbox coordinates
        return self._post_processing(self.default_anchors, feature_maps)

    def _post_processing(self, default_anchors: np.array, feature_maps: list) -> torch.tensor:
        # TODO take list of feature maps, then return (batch size, object number, n_classes + 4) tensor
        batch_size, *_ = feature_maps[0].shape
        cls_score_list = list()
        reg_coord_list = list()
        for f in feature_maps:
            # print("f's shape ",f.shape)
            f = f.permute(0,2,3,1).reshape(batch_size, -1, 4*(self.n_classes+4))
            # print("f transformed shape ", f.shape)
            cls_score = f[:,:,:-4*4].reshape(batch_size, -1, self.n_classes)
            # flatten this tensor is ok, the grid of anchor boxes and tensor is matched
            reg_coord = f[:,:,-4*4:].reshape(batch_size, -1, 4)

            cls_score_list.append(cls_score)
            reg_coord_list.append(reg_coord)

        cls_score = torch.cat(cls_score_list, dim=1)
        reg_coord = torch.cat(reg_coord_list, dim=1)
        # print("cls shape ",cls_score.shape)
        # print("reg shape ",reg_coord.shape)
        reg_coord = reg_coord + torch.tensor(default_anchors, dtype=torch.float).to(reg_coord.device)

        return cls_score, reg_coord

    def _set_default_anchor_boxes(self, featuremap_sizes: list, reduction_ratio: list) -> np.array:
        # TODO set default anchor boxes corresponding to each output feature map
        def generate_anchor(base_grid: tuple, base_interval: tuple) -> np.array:
            """
            Compute default anchor boxes for the baseline grid of each feature map
            Each grid will have 4 boxes
            :param base_grid: tuple (ymin, xmin) center point
            :param base_interval: tuple (y interval, x interval)
            :return: (4,4) shape of numpy array, the order is (small center, horizontal, vertical, large)
            """
            # for broadcasting use numpy
            base_grid = np.array(base_grid)
            base_interval = np.array(base_interval*2)
            # addition, subtract operation will create new numpy array object
            center_coord = np.array([*base_grid, *(base_grid+1)])
            horizon_coord = center_coord + np.array([0, -1, 0, +1])
            vertical_coord = center_coord + np.array([-1, 0, +1, 0])
            large_coord = center_coord + np.array([-1, -1, 1, 1])

            anchors = np.vstack([center_coord, horizon_coord,
                                 vertical_coord, large_coord,])
            anchors = anchors * base_interval

            return anchors

        anchors = []
        # loop for each feature map
        for f_size, r_ratio in zip(featuremap_sizes, reduction_ratio):
            # TODO torch tensor의 reshape 기능으로 flatten을 하면 편한데 그 순서를 맞추기 위해 여기 deafult anchormap도 reshape함수를 써서 flatten 하게 만듬
            # TODO 각각 feature map의 그리드를 매칭시킵시다. -> tensor.permute로 해결해보자
            base_grid_list = [[i,j] for i in range(f_size[0]) for j in range(f_size[1])]
            # loop for each grid cell in the feature map
            for base_grid in base_grid_list:
                anchors.append(generate_anchor(base_grid, r_ratio))

        anchors = np.vstack(anchors)
        return np.clip(anchors, 0, [*self.input_size, *self.input_size])

    def _set_featuremap_sizes(self):
        """

        :return:
        :featuremap_sizes: list of tuple, each tuple contain feature map's h,w size, tuple ordered by largest to smallest feature map
        :reduction_ratio: list of tuple, how the size(h,w) of feature map reduced from input size
        """
        sample_tensor = torch.zeros((1,3,*self.input_size), dtype=torch.float32)
        feature_maps = [self.base_network(sample_tensor)]

        for layer in self.extra_feature_layers:
            feature_maps.append(layer(feature_maps[-1]))

        # remember of reduction ratio compare to original image and feature maps
        reduction_ratio = [
            (torch.tensor(self.input_size, dtype=torch.float32) / torch.tensor(f.shape[-2:], dtype=torch.float32)).tolist()
            for f in feature_maps]

        featuremap_sizes = [list(f.shape)[2:] for f in feature_maps]
        # TODO Complete this function for check size of feature maps
        return featuremap_sizes, reduction_ratio

# TODO need to make loss function, bbox transform matric
def multiboxLoss(x, c : torch.tensor, l : torch.tensor, g, alpha=1):
    # deprecated
    """
    this is for single examples not for minibatch
    :param x: The matching array, which contains each anchor index and gt box index
    :param c: classification result#####, suppose this param is already softmaxed
    :param l: predicted boxes
    :param g: gt bboxes
    :return: ssd multibox loss
    """
    # TODO Complete loss function, need confidence loss, localization loss
    # TODO Transform this for the multi batch
    x = np.argwhere(x == True) # pep 8 warning, using numpy broadcasting
    n = len(x)
    if n == 0:
        return 0

    for i, j in x:
        # localization loss
        loc_loss = smooth_l1_loss(l[i], g[j, 1:], reduction='sum') # g[j,1:] : gt bbox coordinates

    # positive (background loss)
    pos_label = torch.tensor([g[j, 0] for _,j in x]).to(device=g.device, dtype=torch.long) # need to cast long type for classification loss
    pos_cls_loss = nll_loss(log_softmax(c[x[:, 0]], dim=1), pos_label, reduction='sum')

    # negative (background loss)
    neg_idx = np.array([i for i in range(len(l)) if i not in x[:, 0]])
    neg_idx = hard_negative_mining(neg_idx, c)
    neg_label = torch.zeros(len(neg_idx), dtype=torch.long).to(device=g.device)
    neg_cls_loss = nll_loss(log_softmax(c[neg_idx], dim=1), neg_label, reduction='sum')

    return (alpha * loc_loss + neg_cls_loss + pos_cls_loss) / n

def multiboxLoss_batch(x, c : torch.tensor, l : torch.tensor, g, alpha=1):
    """
    Multibox loss, currently support mini-batch processing
    :param x: The matching array, which contains each anchor index and gt box index
    :param c: classification result, suppose this param is already softmaxed
    :param l: predicted boxes
    :param g: gt bboxes
    :return: ssd multibox loss
    """
    cls_labels = x[:, :, 0].to(device=g.device, dtype=torch.long)
    reg_labels = x[:, :, 1:].to(device=g.device, dtype=torch.float)

    masks = torch.cat([(i != 0)[None] for i in cls_labels])
    masks_neg = (masks == False)
    n = masks.to(dtype=torch.int).sum(dim=1)

    # localization loss
    loc_loss = smooth_l1_loss(l[masks], reg_labels[masks], size_average=False, reduction='mean')
    # positive (background loss)
    pos_cls_loss = nll_loss(log_softmax(c[masks], dim=1), cls_labels[masks], size_average=False, reduction='mean',
                            ignore_index=0)

    # negative (background loss)
    neg_idx = hard_negative_mining(masks_neg, c, n.sum())
    neg_label = torch.zeros(len(neg_idx), dtype=torch.long).to(device=g.device)
    neg_cls_loss = nll_loss(log_softmax(c[masks_neg][neg_idx], dim=1), neg_label, size_average=False, reduction='mean')

    return alpha * loc_loss + neg_cls_loss + pos_cls_loss

def hard_negative_mining(neg_masks: torch.tensor, c: torch.tensor, pos_size):
    # TODO Complete this function
    """
    the output of model has highly imbalanced(in terms of class balance). the prediction of background will large numbers,
    so we need to select the some negative(background) outputs for training
    the ratio of pos and neg should be at most 1:3
    :param neg_masks: tensor shape of (batch_size, n_default anchor) which is generated by matching algorithm, for background
    :param c: class score
    :return: selected negative sample indices
    """
    with torch.no_grad():
        bg_score = c[:, :, 0]
        neg_score = bg_score[neg_masks]
        high_idx = neg_score.argsort(dim=0)[:pos_size * 3]
    return high_idx

def ssd_box_matching(default_boxes, gt_boxes):
    """
    The matching strategy in SSD paper

    :param default_boxes: default anchors
    :param gt_boxes: gt bboxes
    :return: set a tensor shape of (len(default_boxes), 5) to contain cls label, reg label
    """
    # TODO SSD object can remember the default matching box indices, it helps to remove duplicates
    # TODO currently brute force way, check the efficiency
    match_flag = torch.zeros((len(default_boxes), 5), dtype=torch.float)
    for d, d_box in enumerate(default_boxes):
        temp = torch.zeros(len(gt_boxes), dtype=torch.float) - 1

        for gt, gt_box in enumerate(gt_boxes[gt_boxes[:,0] != -1]): # -1 in label means no more objects
            iou_score = iou(d_box, gt_box[1:])
            if iou_score > 0.5:
                temp[gt] = iou_score

        max_val, indices = temp.max(dim=0)
        if not max_val == -1: # not background, default box is matching to at least one objects
            match_flag[d] = gt_boxes[indices]

    return match_flag

def ssd_box_matching_batch(default_boxes, gt_boxes_batch):
    """
    box matchong for batch
    :param default_boxes:
    :param gt_boxes_batch:
    :return: to batch
    """
    # TODO Too time consuming.... need to be parallel if possible
    box_lists = [ssd_box_matching(default_boxes, gt_boxes)[None] for gt_boxes in gt_boxes_batch]
    return torch.cat(box_lists, dim=0)
