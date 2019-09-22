
def iou(a, b):
    """
    Intersection over union

    :param a: box a
    :param b: box b
    :return: iou score
    """
    a_y1, a_x1, a_y2, a_x2 = a
    b_y1, b_x1, b_y2, b_x2 = b

    a_area = (a_y2 - a_y1) * (a_x2 - a_x1)
    b_area = (b_y2 - b_y1) * (b_x2 - b_x1)

    inter = (min(a_y2, b_y2) - max(a_y1, b_y1)) * ((min(a_x2, b_x2) - max(a_x1, b_x1)))

    iou = inter / (a_area + b_area - inter)

    return iou

def yxyx_to_yxhw(box):
    y1, x1, y2, x2 = box
    box[0] = (y2 + y1) / 2 # cy
    box[1] = (x2 + x1) / 2 # cx
    box[2] = y2 - y1 # h
    box[3] = x2 - x1 # w
    return box #[cy, cx, h, w]