import numpy as np


def iou(a, b, box_coord='yxhw'):
    """
    Intersection over union

    :param a: box a
    :param b: box b
    :return: iou score
    """
    assert box_coord in ['yxhw', 'yxyx']
    if box_coord == 'yxhw':
        a_yc, a_xc, a_h, a_w = a
        b_yc, b_xc, b_h, b_w = b

        a_area = a_h * a_w
        b_area = b_h * b_w

    else:
        a_y1, a_x1, a_y2, a_x2 = a
        b_y1, b_x1, b_y2, b_x2 = b

        a_area = (a_y2 - a_y1) * (a_x2 - a_x1)
        b_area = (b_y2 - b_y1) * (b_x2 - b_x1)

        inter = (min(a_y2, b_y2) - max(a_y1, b_y1)) * (min(a_x2, b_x2) - max(a_x1, b_x1))

        return inter / (a_area + b_area - inter)


def yxyx_to_yxhw(box):  # need to comment
    y1, x1, y2, x2 = box
    box[0] = (y2 + y1) / 2  # cy
    box[1] = (x2 + x1) / 2  # cx
    box[2] = y2 - y1  # h
    box[3] = x2 - x1  # w
    return box  # [cy, cx, h, w]


def yxhw_to_yxyx(box):  # need to comment
    yc, xc, h, w = box
    box[0] = yc - (h / 2)  # y1
    box[1] = xc - (w / 2)  # x1
    box[2] = yc + (h / 2)  # y2
    box[3] = xc + (w / 2)  # x2
    return box  # [y1, x1, y2, x2]


def nms():
    # TODO complete nms(non maximum suppression) algorithm
    return 0


def confusion_matrix(prediction, label, classes):
    """
    Sci-kit learn style confusion matrix
    :param prediction:
    :param label:
    :param classes:
    :return:
    """
    # TODO need to mapping function, input of classes is not restricted. using dictionary helps to make the mapping function
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int)

    for pred, gt in zip(prediction, label):
        confusion[gt, pred] += 1

    return confusion


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

if __name__=='__main__':
    prediction = np.array([1,2,1,0,3,3,1,2,0,1,4], dtype=np.int)
    label = np.array([1,0,1,0,3,3,1,2,2,1,0], dtype=np.int)
    print(confusion_matrix(prediction, label, classes=[0,1,2,3,4]))
