import numpy as np
from misc.utils import iou

"""
Scoring metrics for detection problem
"""
def ap_detection(pred_boxes, gt_boxes, iou_thre=0.5):
    iou_score_box = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float)
    for p, pbox in enumerate(pred_boxes):
        for g, gbox in enumerate(gt_boxes):
            iou_score_box[p, g] = iou(pbox, gbox)

    return 0