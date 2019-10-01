
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

def yxyx_to_yxhw(box):
    y1, x1, y2, x2 = box
    box[0] = (y2 + y1) / 2  # cy
    box[1] = (x2 + x1) / 2  # cx
    box[2] = y2 - y1  # h
    box[3] = x2 - x1  # w
    return box  # [cy, cx, h, w]

def yxhw_to_yxyx(box):
    yc, xc, h, w = box
    box[0] = yc - (h / 2)  # y1
    box[1] = xc - (w / 2)  # x1
    box[2] = yc + (h / 2)  # y2
    box[3] = xc + (w / 2)  # x2
    return box  # [y1, x1, y2, x2]

def nms():
    # TODO complete nms(non maximum suppression) algorithm
    return 0