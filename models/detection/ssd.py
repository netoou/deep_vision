"""
Single Shot Multi-box detector
"""
import torch
import torchvision

from torch import nn
from torch.nn import Module

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
            print("f's shape ",f.shape)
            f = f.permute(0,2,3,1).reshape(batch_size, -1, 4*(self.n_classes+4))
            print("f transformed shape ", f.shape)
            cls_score = f[:,:,:-4*4].reshape(batch_size, -1, self.n_classes)
            # flatten this tensor is ok, the grid of anchor boxes and tensor is matched
            reg_coord = f[:,:,-4*4:].reshape(batch_size, -1, 4)

            cls_score_list.append(cls_score)
            reg_coord_list.append(reg_coord)

        cls_score = torch.cat(cls_score_list, dim=1)
        reg_coord = torch.cat(reg_coord_list, dim=1)
        print("cls shape ",cls_score.shape)
        print("reg shape ",reg_coord.shape)
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


if __name__=='__main__':
    a = np.array([1,2,3,])
    b = np.array([1,2,3,])
    print(np.vstack([a,b])*np.array([-1,-2,-3]))




