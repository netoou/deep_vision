import numpy as np
from misc.utils import iou, sigmoid, confusion_matrix

import torch

"""
Scoring metrics for detection problem
"""
def map_detection(pred_cls, pred_boxes, gt_cls, gt_boxes, iou_thre=0.5):
    """

    :param pred_cls: class score, shape of (number of samples, number of classes)
    :param pred_boxes:  predicted bbox, shape of (n of samples, 4)
    :param gt_cls: ground truth class label, shape (n of samples)
    :param gt_boxes: ground truth bbox coord, shape (n of samples, 4)
    :param iou_thre: coef for filtering undesirable predictions
    :return: average precision score
    """
    pred_cls = np.array(pred_cls)
    pred_boxes = np.array(pred_boxes)
    gt_cls = np.array(gt_cls)
    gt_boxes = np.array(gt_boxes)

    classes = np.unique(gt_cls)

    iou_score_box = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float)
    for p, pbox in enumerate(pred_boxes):
        for g, gbox in enumerate(gt_boxes):
            iou_score_box[p, g] = iou(pbox, gbox)

    # TODO need to seperate array to each class
    alive_box_indices = np.argwhere(iou_score_box > iou_thre)
    # now we have the indices of alive boxes(iou over iou_threshold), [[pred_idx, gt_idx]] <--- 2d matrix
    label_matched_pred_score = np.zeros((len(alive_box_indices)), dtype=np.float)
    # suppose there is one prediction per one predicted box, class

    pred_cls = pred_cls[alive_box_indices[:,0]]
    gt_cls = pred_cls[alive_box_indices[:,1]]

    sum_ap = 0.0
    for cls in classes:
        mask_cls = gt_cls == cls
        sum_ap += average_precision(pred_cls[mask_cls], gt_cls[mask_cls])

    return sum_ap / len(classes)

def precision_recall(pred_score: np.array, gt_label: np.array):
    """
    Average precision function, the elements of two arrays are aligned with same index

    Precision: TP / (TP + FP)
    Recall: TP / (TP + FN)

    :param pred_score: prediction score of specific class
    :param gt_label: ground truth class of predicted score, ay all 1 for detection problem
    :return: average precision, precision list, decision thresholds
    """
    assert len(np.unique(gt_label)) < 3, "Too many labels, 2 classes only"
    # numpy array is mutable object, need to create new object with same values
    pred_score = np.array(pred_score)
    gt_label = np.array(gt_label)

    gt_label = np.clip(gt_label, 0, 1)  # clip to binary case
    sorting_idx = np.argsort(pred_score)
    pred_score = pred_score[sorting_idx]
    gt_label = gt_label[sorting_idx]

    decision_thresholds = np.unique(pred_score[sorting_idx])[1:]
    precisions = np.ones(len(decision_thresholds) + 1, dtype=np.float)
    recalls = np.zeros(len(decision_thresholds) + 1, dtype=np.float)

    for idx, thres in enumerate(decision_thresholds):
        thred_score = np.array(pred_score >= thres, dtype=np.long)  # machine decision of a point of thresholding coeff
        _, fp, fn, tp = confusion_matrix(thred_score, gt_label, classes=[0, gt_label.max()]).ravel()
        precisions[idx] = tp / (tp + fp)
        recalls[idx] = tp / (tp + fn)

    return precisions, recalls, decision_thresholds

def average_precision(pred_score: np.array, gt_label: np.array):
    """
    Average precision
    :param pred_score:
    :param gt_label:
    :return:
    """
    precision, recall, _ = precision_recall(pred_score, gt_label)
    precision = np.flip(precision, axis=0)
    recall = np.flip(recall, axis=0)
    diff = np.diff(recall)
    ap_score = np.sum(diff * precision[1:])

    return ap_score


if __name__ == '__main__':
    print(precision_recall([0.1, 0.9, 0.35, 0.8, 0.75, 0.5, 0.33, 0.6], [0,0,1,1,0,1,1,1]))  # precision recall testcase
    print(average_precision([0.1, 0.9, 0.35, 0.8, 0.75, 0.5, 0.33, 0.6], [0,0,1,1,0,1,1,1]))


