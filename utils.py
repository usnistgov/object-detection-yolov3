# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def reorg_layer_np(feature_map, stride, n_classes, anchors):
    # feature_map is [N, C, H, W]

    batchsize = feature_map.shape[0]
    n_anchors = len(anchors)
    grid_size = feature_map.shape[2:]
    n_ch = 5 + n_classes

    # Convert feature_map to [N, H, W, C]
    feature_map = np.transpose(feature_map, (0, 2, 3, 1))
    # expand out channels dimension
    feature_map = np.reshape(feature_map, (batchsize, grid_size[0], grid_size[1], n_anchors, n_ch))
    # feature_map shape = [batch_size, h/stride, w/stride, n_anchors, n_ch]
    # n_ch = [x, y, w, h, objectness_logits, class_logits(one per class)]

    # separate out feature_map components
    boxes = feature_map[..., 0:4]
    objectness_logits = feature_map[..., 4:5]
    class_logits = feature_map[..., 5:]

    x_offset = np.arange(grid_size[1], dtype=np.float32)
    x_offset = np.reshape(x_offset, (1, np.size(x_offset), 1))
    x_offset = np.broadcast_to(x_offset, [grid_size[0], grid_size[1], n_anchors])
    x_offset = np.reshape(x_offset, (1, x_offset.shape[0], x_offset.shape[1], x_offset.shape[2]))

    y_offset = np.arange(grid_size[0], dtype=np.float32)
    y_offset = np.reshape(y_offset, (np.size(y_offset), 1, 1))
    y_offset = np.broadcast_to(y_offset, [grid_size[0], grid_size[1], n_anchors])
    y_offset = np.reshape(y_offset, (1, y_offset.shape[0], y_offset.shape[1], y_offset.shape[2]))

    # (c_x, c_y) the cell offsets is the integer number of the cell times the stride

    # b_x = cell_size * sigmoid(t_x) + c_x
    # b_y = cell_size * sigmoid(t_x) + c_y
    # b_w = p_w * exp(t_w)
    # b_h = p_h * exp(t_h)

    # perform conversion from (t_x, t_y) to (b_x, b_y)
    # box_xy = cell_size * sigmoid(xy) + cell_offset
    # box_xy = cell_size * sigmoid(xy) + cell_size * xy_offset
    # box_xy = cell_size * (sigmoid(xy) + xy_offset)
    boxes[..., 0] = (sigmoid(boxes[..., 0]) + x_offset) * stride
    boxes[..., 1] = (sigmoid(boxes[..., 1]) + y_offset) * stride

    anchors = np.array(anchors)
    w_anchors = np.reshape(anchors[:, 0], (1, 1, 1, n_anchors))
    h_anchors = np.reshape(anchors[:, 1], (1, 1, 1, n_anchors))

    boxes[..., 2] = np.exp(boxes[..., 2]) * w_anchors
    boxes[..., 3] = np.exp(boxes[..., 3]) * h_anchors

    # boxes = torch.cat((box_xy, box_wh), dim=-1)
    # boxes are [x, y, h, w] with (x, y) being center

    objectness_logits = sigmoid(objectness_logits)
    class_logits = softmax(class_logits, axis=-1)
    predictions = np.concatenate((boxes, objectness_logits, class_logits), axis=-1)
    predictions = np.reshape(predictions, (batchsize, -1, n_ch))
    # move (xc, yc) to (x, y)
    predictions[:, :, 0] = predictions[:, :, 0] - (predictions[:, :, 2] / 2)
    predictions[:, :, 1] = predictions[:, :, 1] - (predictions[:, :, 3] / 2)
    #  prediction contains [x, y, w, h] with (x, y) being the top left of the box

    return predictions


# def reorg_layer_postprocess(feature_map, image_size, stride, n_classes, anchors):
#     # feature_map is [N, C, H, W]
#
#     batchsize = feature_map.shape[0]
#     n_anchors = len(anchors)
#     # grid_size = feature_map.shape[2:]
#     grid_size = (int(image_size[0] / stride), int(image_size[1] / stride))
#     n_ch = 5 + n_classes
#     # dtype = torch.cuda.FloatTensor if feature_map.is_cuda else torch.FloatTensor
#
#     # Convert feature_map to [N, H, W, C]
#     feature_map = feature_map.permute(0, 2, 3, 1)
#     # expand out channels dimension
#     feature_map = feature_map.view(batchsize, grid_size[0], grid_size[1], n_anchors, n_ch)
#     # feature_map shape = [batch_size, h/stride, w/stride, n_anchors, n_ch]
#
#     # feature_map.shape() = [B, nb_anchors, fsize_h, fsize_w, n_ch]
#     # n_ch = [x, y, w, h, objectness_logits, class_logits(one per class)]
#
#     # separate out feature_map components
#     boxes = feature_map[..., 0:4]
#     # box_xy = feature_map[..., 0:2]
#     # box_wh = feature_map[..., 2:4]
#     objectness_logits = feature_map[..., 4:5]
#     class_logits = feature_map[..., 5:]
#
#     x_offset = np.arange(grid_size[1], dtype=np.float32)
#     # x_offset = x_offset.reshape((1, 1, -1, 1))
#     # x_offset = np.broadcast_to(x_offset, feature_map.shape[0:4])
#     x_offset = np.reshape(x_offset, (1, np.size(x_offset), 1))
#     x_offset = np.broadcast_to(x_offset, [grid_size[0], grid_size[1], n_anchors])
#     x_offset = np.reshape(x_offset, (1, x_offset.shape[0], x_offset.shape[1], x_offset.shape[2]))
#     # x_offset = dtype(x_offset)
#     x_offset = torch.from_numpy(x_offset)
#     if feature_map.is_cuda:
#         x_offset = x_offset.cuda()
#
#     y_offset = np.arange(grid_size[0], dtype=np.float32)
#     # y_offset = y_offset.reshape((1, -1, 1, 1))
#     # y_offset = np.broadcast_to(y_offset, feature_map.shape[0:4])
#     y_offset = np.reshape(y_offset, (np.size(y_offset), 1, 1))
#     y_offset = np.broadcast_to(y_offset, [grid_size[0], grid_size[1], n_anchors])
#     y_offset = np.reshape(y_offset, (1, y_offset.shape[0], y_offset.shape[1], y_offset.shape[2]))
#     # y_offset = dtype(y_offset)
#     y_offset = torch.from_numpy(y_offset)
#     if feature_map.is_cuda:
#         y_offset = y_offset.cuda()
#
#     # (c_x, c_y) the cell offsets is the integer number of the cell times the stride
#
#     # b_x = cell_size * sigmoid(t_x) + c_x
#     # b_y = cell_size * sigmoid(t_x) + c_y
#     # b_w = p_w * exp(t_w)
#     # b_h = p_h * exp(t_h)
#
#     # perform conversion from (t_x, t_y) to (b_x, b_y)
#     # box_xy = cell_size * sigmoid(xy) + cell_offset
#     # box_xy = cell_size * sigmoid(xy) + cell_size * xy_offset
#     # box_xy = cell_size * (sigmoid(xy) + xy_offset)
#     # box_xy[..., 0] = (torch.sigmoid(box_xy[..., 0]) + x_offset) * self.stride
#     # box_xy[..., 1] = (torch.sigmoid(box_xy[..., 1]) + y_offset) * self.stride
#     boxes[..., 0] = (torch.sigmoid(boxes[..., 0]) + x_offset) * stride
#     boxes[..., 1] = (torch.sigmoid(boxes[..., 1]) + y_offset) * stride
#
#     anchors = np.array(anchors)
#     w_anchors = np.reshape(anchors[:, 0], (1, 1, 1, n_anchors))
#     h_anchors = np.reshape(anchors[:, 1], (1, 1, 1, n_anchors))
#     # w_anchors = dtype(w_anchors)
#     # h_anchors = dtype(h_anchors)
#     w_anchors = torch.from_numpy(w_anchors)
#     h_anchors = torch.from_numpy(h_anchors)
#     if feature_map.is_cuda:
#         w_anchors = w_anchors.cuda()
#         h_anchors = h_anchors.cuda()
#
#     # w_anchors = dtype(np.broadcast_to(np.reshape(anchors[:, 0], (1, 1, 1, self.n_anchors)), feature_map.shape[:4]))
#     # h_anchors = dtype(np.broadcast_to(np.reshape(anchors[:, 1], (1, 1, 1, self.n_anchors)), feature_map.shape[:4]))
#
#     # box_wh[..., 0] = torch.exp(box_wh[..., 0]) * w_anchors
#     # box_wh[..., 1] = torch.exp(box_wh[..., 1]) * h_anchors
#     boxes[..., 2] = torch.exp(boxes[..., 2]) * w_anchors
#     boxes[..., 3] = torch.exp(boxes[..., 3]) * h_anchors
#
#     # boxes = torch.cat((box_xy, box_wh), dim=-1)
#     # boxes are [x, y, h, w] with (x, y) being center
#
#     objectness_logits = torch.sigmoid(objectness_logits)
#     class_logits = torch.nn.Softmax(dim=-1)(class_logits)
#     predictions = torch.cat((boxes, objectness_logits, class_logits), dim=-1)
#     # predictions = predictions.view(batchsize, -1, n_ch)
#     n = predictions.shape[1] * predictions.shape[2]
#     predictions = predictions.view(batchsize, n, n_ch)
#     # move (xc, yc) to (x, y)
#     predictions[:, :, 0] = predictions[:, :, 0] - (predictions[:, :, 2] / 2)
#     predictions[:, :, 1] = predictions[:, :, 1] - (predictions[:, :, 3] / 2)
#     #  prediction contains [x, y, w, h] with (x, y) being the top left of the box
#
#     return predictions


def postprocess_numpy(prediction, num_classes, score_threshold=0.1, iou_threshold=0.3, min_box_size=12):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (numpy array): The shape is :math:`(N, B, 5+nb_classes)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`x, y, w, h` where `x` and `y` represent the top left of the box
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

    output = [np.zeros((0, 6)) for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # remove detections smaller than min box size
        w = image_pred[..., 2]
        h = image_pred[..., 3]
        idx = (w > min_box_size) & (h > min_box_size)
        image_pred = image_pred[idx]

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue

        # Get detections with higher confidence scores than the threshold
        class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=1)
        class_prob = np.amax(image_pred[:, 5:5 + num_classes], axis=1)
        scores = np.sqrt(image_pred[:, 4] * class_prob)  # sqrt undoes the objectness * class prob effective squaring
        idx = (scores >= score_threshold).squeeze().nonzero()[0]

        # If none are remaining => process next image
        if not idx.shape[0]:
            continue

        boxes = image_pred[idx, :4].squeeze()
        scores = scores[idx].squeeze()
        class_pred = class_pred[idx].squeeze()

        # Iterate through all predicted classes
        unique_labels = np.unique(class_pred)
        for c in unique_labels:
            # Get the detections with the particular class
            boxes_class = boxes[class_pred == c, :]
            scores_class = scores[class_pred == c]
            pred_class = class_pred[class_pred == c]

            # convert [x, y, w, h] into [x1, y1, x2, y2]
            boxes_xyxy = boxes_class.copy()
            boxes_xyxy[:, 0] = boxes_class[:, 0]
            boxes_xyxy[:, 1] = boxes_class[:, 1]
            boxes_xyxy[:, 2] = boxes_class[:, 0] + boxes_class[:, 2]
            boxes_xyxy[:, 3] = boxes_class[:, 1] + boxes_class[:, 3]

            nms_out_index = single_class_nms(boxes_xyxy[:, :4], scores_class, iou_threshold)
            boxes_class = boxes_class[nms_out_index, :]
            scores_class = scores_class[nms_out_index].astype(np.float32).reshape(-1, 1)
            pred_class = pred_class[nms_out_index].astype(np.float32).reshape(-1, 1)
            results = np.concatenate((boxes_class, scores_class, pred_class), axis=-1)

            output[i] = np.concatenate((output[i], results))

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





