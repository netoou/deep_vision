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

    iou_score_box = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float)
    for p, pbox in enumerate(pred_boxes):
        for g, gbox in enumerate(gt_boxes):
            iou_score_box[p, g] = iou(pbox, gbox)

    # TODO need to seperate array to each class
    alive_box_indices = np.argwhere(iou_score_box > iou_thre)
    # now we have the indices of alive boxes(iou over iou_threshold), [[pred_idx, gt_idx]] <--- 2d matrix
    label_matched_pred_score = np.zeros((len(alive_box_indices)), dtype=np.float)
    # suppose there is one prediction per one predicted box, class
    for idx, (p, g) in enumerate(alive_box_indices):
        gt = gt_cls[g]
        pred = sigmoid(pred_cls[p])
        label_matched_pred_score[idx] = pred

    n_alive_sample = len(label_matched_pred_score)
    # fitting to order
    sorting_idx = np.argsort(label_matched_pred_score)
    decision_threshold = np.unique(label_matched_pred_score[sorting_idx])

    for thres in decision_threshold:
        thred_score = np.array(label_matched_pred_score > thres, dtype=np.long)

    return decision_threshold

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
    precision, recall, _ = precision_recall(pred_score, gt_label)
    precision = np.flip(precision, axis=0)
    recall = np.flip(recall, axis=0)
    diff = np.diff(recall)
    ap_score = 0.0
    # p_max = 1.0
    #     # r_point = 0.0
    #     # for i, (p, r) in enumerate(zip(precision, recall)):  # smoothing p-r curve
    #     #     if p < p_max:
    #     #         ap_score += p_max * (r - r_point)
    #     #         p_max = precision[i+1:].max()
    #     #         r_point = r

    # ap_score += p_max * (1 - r_point)


    return ap_score


if __name__ == '__main__':
    print(precision_recall([0.1, 0.9, 0.35, 0.8, 0.75, 0.5, 0.33, 0.6], [0,0,1,1,0,1,1,1]))  # precision recall testcase
    print(average_precision([0.1, 0.9, 0.35, 0.8, 0.75, 0.5, 0.33, 0.6], [0,0,1,1,0,1,1,1]))


