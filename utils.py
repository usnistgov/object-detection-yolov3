import torch
import numpy as np


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(prediction, num_classes, score_threshold=0.1, iou_threshold=0.3, min_box_size=12):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 5+nb_classes)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent the center of the box
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        score_threshold (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        iou_threshold (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    # move (xc, yc) to (x, y)
    prediction[:, :, 0] = prediction[:, :, 0] - (prediction[:, :, 2] / 2)
    prediction[:, :, 1] = prediction[:, :, 1] - (prediction[:, :, 3] / 2)
    #  prediction contains [x, y, w, h] with (x, y) being the top left of the box

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_prob, _ = torch.max(image_pred[:, 5:5 + num_classes], dim=1)
        scores = torch.sqrt(image_pred[:, 4] * class_prob)  # sqrt undoes the objectness * class prob effective squaring
        conf_mask = (scores >= score_threshold).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # remove detections smaller than min box size
        w = image_pred[..., 2]
        h = image_pred[..., 3]
        idx = (w > min_box_size) & (h > min_box_size)
        image_pred = image_pred[idx]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get detections with higher confidence scores than the threshold
        class_prob, class_pred = torch.max(image_pred[:, 5:5 + num_classes], dim=1)
        scores = torch.sqrt(image_pred[:, 4] * class_prob)  # sqrt undoes the objectness * class prob effective squaring
        idx = (scores >= score_threshold).squeeze().nonzero()
        boxes = image_pred[idx, :4].squeeze()
        class_pred = class_pred[idx].squeeze()

        # Iterate through all predicted classes
        unique_labels = class_pred.cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            boxes_class = boxes[class_pred == c, :]
            scores_class = scores[class_pred == c]
            pred_class = class_pred[class_pred == c]

            # convert [x, y, w, h] into [x1, y1, x2, y2]
            boxes_xyxy = boxes_class.new(boxes_class.shape)
            boxes_xyxy[:, 0] = boxes_class[:, 0]
            boxes_xyxy[:, 1] = boxes_class[:, 1]
            boxes_xyxy[:, 2] = boxes_class[:, 0] + boxes_class[:, 2]
            boxes_xyxy[:, 3] = boxes_class[:, 1] + boxes_class[:, 3]

            nms_in = boxes_xyxy.cpu().numpy()
            nms_scores = scores_class.cpu().numpy()
            nms_out_index = nms(nms_in[:, :4], iou_threshold, score=nms_scores)
            boxes_class = boxes_class[nms_out_index, :]
            scores_class = scores_class[nms_out_index].to(torch.float32).reshape(-1, 1)
            pred_class = pred_class[nms_out_index].to(torch.float32).reshape(-1, 1)
            results = torch.cat((boxes_class, scores_class, pred_class), dim=-1)
            if output[i] is None:
                output[i] = results
            else:
                output[i] = torch.cat((output[i], results))

    # [x, y, w, h, score, pred_class]
    return output


def bboxes_iou_xywh_broadcast(true_boxes, pred_boxes):
    # shape:
    # true_boxes: [V, 4]
    # pred_boxes: [grid_size, grid_size, num_anchors, 4]
    # V = number of valid ground truth boxes

    if true_boxes.shape[-1] != 4 or len(true_boxes.shape) != 2 or pred_boxes.shape[-1] != 4 or len(pred_boxes.shape) != 4:
        raise IndexError

    V = true_boxes.shape[0]

    # shape: [grid_size, grid_size, num_anchors, 1, 2]
    pred_boxes = pred_boxes.unsqueeze(dim=-2)

    # shape: [1, V, 2]
    true_boxes = true_boxes.unsqueeze(dim=0)

    # boxes are [x, y, h, w] with (x, y) being center

    # [grid_size, grid_size, num_anchors, 1, 2] & [1, V, 2] ==> [grid_size, grid_size, num_anchors, V, 2]
    intersect_mins = torch.max(pred_boxes[..., 0:2] - (pred_boxes[..., 2:4] / 2), true_boxes[..., 0:2] - (true_boxes[..., 2:4] / 2))
    intersect_maxs = torch.min(pred_boxes[..., 0:2] + (pred_boxes[..., 2:4] / 2), true_boxes[..., 0:2] + (true_boxes[..., 2:4] / 2))

    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))

    # shape: [grid_size, grid_size, num_anchors, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [grid_size, grid_size, num_anchors, 1]
    pred_box_area = pred_boxes[..., 2] * pred_boxes[..., 3]
    # shape: [1, V]
    true_box_area = true_boxes[..., 2] * true_boxes[..., 3]
    # [grid_size, grid_size, num_anchors, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area)
    return iou


def bboxes_iou_xywh(bboxes_a, bboxes_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
    # bottom right
    br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

    area_a = torch.prod(bboxes_a[:, 2:], 1)
    area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def bboxes_iou_xyxy(bboxes_a, bboxes_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
    # bottom right
    br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


# def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
#     """Calculate the Intersection of Unions (IoUs) between bounding boxes.
#     IoU is calculated as a ratio of area of the intersection
#     and area of the union.
#
#     Args:
#         bbox_a (array): An array whose shape is :math:`(N, 4)`.
#             :math:`N` is the number of bounding boxes.
#             The dtype should be :obj:`numpy.float32`.
#         bbox_b (array): An array similar to :obj:`bbox_a`,
#             whose shape is :math:`(K, 4)`.
#             The dtype should be :obj:`numpy.float32`.
#     Returns:
#         array:
#         An array whose shape is :math:`(N, K)`. \
#         An element at index :math:`(n, k)` contains IoUs between \
#         :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
#         box in :obj:`bbox_b`.
#
#     from: https://github.com/chainer/chainercv
#     """
#     if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
#         raise IndexError
#
#     # top left
#     if xyxy:
#         tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
#         # bottom right
#         br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
#         area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
#         area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
#     else:
#         tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
#                         (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
#         # bottom right
#         br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
#                         (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
#
#         area_a = torch.prod(bboxes_a[:, 2:], 1)
#         area_b = torch.prod(bboxes_b[:, 2:], 1)
#     en = (tl < br).type(tl.type()).prod(dim=2)
#     area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
#     return area_i / (area_a[:, None] + area_b - area_i)

#
# def label2yolobox(labels, info_img, maxsize, lrflip):
#     """
#     Transform coco labels to yolo box labels
#     Args:
#         labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
#             Each label consists of [class, x, y, w, h] where \
#                 class (float): class index.
#                 x, y, w, h (float) : coordinates of \
#                     left-top points, width, and height of a bounding box.
#                     Values range from 0 to width or height of the image.
#         info_img : tuple of h, w, nh, nw, dx, dy.
#             h, w (int): original shape of the image
#             nh, nw (int): shape of the resized image without padding
#             dx, dy (int): pad size
#         maxsize (int): target image size after pre-processing
#         lrflip (bool): horizontal flip flag
#
#     Returns:
#         labels:label data whose size is :math:`(N, 5)`.
#             Each label consists of [class, xc, yc, w, h] where
#                 class (float): class index.
#                 xc, yc (float) : center of bbox whose values range from 0 to 1.
#                 w, h (float) : size of bbox whose values range from 0 to 1.
#     """
#     h, w, nh, nw, dx, dy = info_img
#     x1 = labels[:, 1] / w
#     y1 = labels[:, 2] / h
#     x2 = (labels[:, 1] + labels[:, 3]) / w
#     y2 = (labels[:, 2] + labels[:, 4]) / h
#     labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
#     labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
#     labels[:, 3] *= nw / w / maxsize
#     labels[:, 4] *= nh / h / maxsize
#     if lrflip:
#         labels[:, 1] = 1 - labels[:, 1]
#     return labels
#
#
# def yolobox2label(box, info_img):
#     """
#     Transform yolo box labels to yxyx box labels.
#     Args:
#         box (list): box data with the format of [yc, xc, w, h]
#             in the coordinate system after pre-processing.
#         info_img : tuple of h, w, nh, nw, dx, dy.
#             h, w (int): original shape of the image
#             nh, nw (int): shape of the resized image without padding
#             dx, dy (int): pad size
#     Returns:
#         label (list): box data with the format of [y1, x1, y2, x2]
#             in the coordinate system of the input image.
#     """
#     h, w, nh, nw, dx, dy = info_img
#     y1, x1, y2, x2 = box
#     box_h = ((y2 - y1) / nh) * h
#     box_w = ((x2 - x1) / nw) * w
#     y1 = ((y1 - dy) / nh) * h
#     x1 = ((x1 - dx) / nw) * w
#     label = [y1, x1, y1 + box_h, x1 + box_w]
#     return label





