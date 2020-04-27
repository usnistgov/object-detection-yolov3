import torch
import numpy as np


def compute_iou(box, boxes, box_area=None, boxes_area=None):
    # this is the iou of the box against all other boxes
    x_left = np.maximum(box[0], boxes[:, 0])
    y_top = np.maximum(box[1], boxes[:, 1])
    x_right = np.minimum(box[2], boxes[:, 2])
    y_bottom = np.minimum(box[3], boxes[:, 3])

    intersections = np.maximum(y_bottom - y_top, 0) * np.maximum(x_right - x_left, 0)
    if box_area is None:
        box_area = (box[2] - box[0]) * (box[3] - box[1])
    if boxes_area is None:
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unions = box_area + boxes_area - intersections
    ious = intersections / unions
    return ious


def single_class_nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        order = order[1:]

        iou = compute_iou(boxes[i, :], boxes[order, :], areas[i], areas[order])

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds]

    return keep


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

        # If none are remaining => process next image
        if not idx.size(0):
            continue

        boxes = image_pred[idx, :4].squeeze()
        scores = scores[idx].squeeze()
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
            nms_out_index = single_class_nms(nms_in[:, :4], nms_scores, iou_threshold)
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

