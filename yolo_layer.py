import torch
import torch.nn
import numpy as np

import utils


class YOLOLayer(torch.nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, config_model, layer_nb, in_ch, ignore_thres=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_nb (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8]  # fixed
        self.anchors = config_model['anchors']
        self.n_anchors = len(self.anchors)
        self.n_classes = config_model['number_classes']
        self.image_size = config_model['image_size']
        self.ignore_thres = ignore_thres
        self.stride = strides[layer_nb]
        # self.l2_loss = torch.nn.MSELoss(size_average=False)
        # self.bce_loss = torch.nn.BCELoss(size_average=False)
        self.layer_nb = layer_nb

        # self.all_anchors_grid = [(w / self.stride, h / self.stride)
        #                          for w, h in self.anchors]
        # self.masked_anchors = [self.all_anchors_grid[i]
        #                        for i in self.anch_mask]
        # self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        # self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        # self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = torch.nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)


    def reorg_layer(self, feature_map):
        # feature_map is [N, C, H, W]

        batchsize = feature_map.shape[0]
        grid_size = feature_map.shape[2:]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if feature_map.is_cuda else torch.FloatTensor

        # Convert feature_map to [N, H, W, C]
        feature_map = feature_map.permute(0, 2, 3, 1)
        # expand out channels dimension
        feature_map = feature_map.view(batchsize, grid_size[0], grid_size[1], self.n_anchors, n_ch)

        # feature_map.shape() = [B, nb_anchors, fsize_h, fsize_w, n_ch]
        # n_ch = [x, y, w, h, objectness_logits, class_logits(one per class)]

        # separate out feature_map components
        box_xy = feature_map[..., 0:2]
        box_wh = feature_map[..., 2:4]
        objectness_logits = feature_map[..., 4:5]
        class_logits = feature_map[..., 5:]

        x_offset = np.arange(grid_size[1], dtype=np.float32)
        x_offset = x_offset.reshape((1, 1, -1, 1))
        x_offset = np.broadcast_to(x_offset, feature_map.shape[0:4])
        x_offset = dtype(x_offset)

        y_offset = np.arange(grid_size[0], dtype=np.float32)
        y_offset = y_offset.reshape((1, -1, 1, 1))
        y_offset = np.broadcast_to(y_offset, feature_map.shape[0:4])
        y_offset = dtype(y_offset)

        # (c_x, c_y) the cell offsets is the integer number of the cell times the stride

        # b_x = cell_size * sigmoid(t_x) + c_x
        # b_y = cell_size * sigmoid(t_x) + c_y
        # b_w = p_w * exp(t_w)
        # b_h = p_h * exp(t_h)

        # perform conversion from (t_x, t_y) to (b_x, b_y)
        # box_xy = cell_size * sigmoid(xy) + cell_offset
        # box_xy = cell_size * sigmoid(xy) + cell_size * xy_offset
        # box_xy = cell_size * (sigmoid(xy) + xy_offset)
        box_xy[..., 0] = (torch.sigmoid(box_xy[..., 0]) + x_offset) * self.stride
        box_xy[..., 1] = (torch.sigmoid(box_xy[..., 1]) + y_offset) * self.stride

        anchors = np.array(self.anchors)
        w_anchors = dtype(np.broadcast_to(np.reshape(anchors[:, 0], (1, 1, 1, self.n_anchors)), feature_map.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(anchors[:, 1], (1, 1, 1, self.n_anchors)), feature_map.shape[:4]))

        box_wh[..., 0] = torch.exp(box_wh[..., 0]) * w_anchors
        box_wh[..., 1] = torch.exp(box_wh[..., 1]) * h_anchors

        boxes = torch.cat((box_xy, box_wh), dim=-1)
        # boxes are [x, y, h, w] with (x, y) being center

        return boxes, objectness_logits, class_logits, x_offset, y_offset

    def gt_boxes_to_feature_map(self, boxes, grid_size):
        # boxes is [x, y, w, h, c] of shape [B, K, 5] B = batch size, K = number boxes
        batch_size = boxes.shape[0]

        label = np.zeros((batch_size, grid_size[0], grid_size[1], self.n_anchors, (5 + self.n_classes)), dtype=np.float32)

        if boxes is None:
            return label

        boxes = boxes.cpu().data.numpy()
        boxes = boxes.astype(np.float32)

        all_batch_boxes = boxes
        anchors = np.asarray(self.anchors, dtype=np.float32)

        for b in range(batch_size):
            idx = np.sum(all_batch_boxes[b,:], axis=-1) != 0
            boxes = all_batch_boxes[b, idx, :]

            if boxes.shape[0] == 0:
                continue

            box_xy = boxes[:, 0:2]
            box_wh = boxes[:, 2:4]

            # move box x,y to middle from upper left
            box_xy = np.floor(box_xy + ((box_wh - 1) / 2.0))
            boxes[:, 0:2] = box_xy
            # boxes is [x, y, w, h] where (x, y) is the center of the box

            anchors_max = anchors / 2.0
            anchors_min = -anchors_max
            # set the center of all boxes as the origin of their coordinates
            # and correct their coordinates
            box_wh = np.expand_dims(box_wh, -2)
            boxes_max = box_wh / 2.0
            boxes_min = -boxes_max

            intersect_mins = np.maximum(boxes_min, anchors_min)
            intersect_maxs = np.minimum(boxes_max, anchors_max)
            intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.0)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = box_wh[..., 0] * box_wh[..., 1]

            anchor_area = anchors[:, 0] * anchors[:, 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # Find best anchor for each true box

            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                i = np.floor(boxes[t, 1] / self.image_size[0] * grid_size[0]).astype('int32')
                j = np.floor(boxes[t, 0] / self.image_size[1] * grid_size[1]).astype('int32')

                c = boxes[t, 4].astype('int32')

                # first dimension is the batch
                label[b, i, j, n, 0:4] = boxes[t, 0:4]
                label[b, i, j, n, 4] = 1.0
                label[b, i, j, n, 5 + c] = 1.0

        return label

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        feature_map = self.conv(xin)
        batchsize = feature_map.shape[0]
        grid_size = feature_map.shape[2:]
        n_ch = 5 + self.n_classes

        batch_pred_boxes, batch_objectness_logits, batch_class_logits, x_offset, y_offset = self.reorg_layer(feature_map)
        # batch_pred_boxes = batch_pred_boxes.cpu()
        # batch_objectness_logits = batch_objectness_logits.cpu()
        # batch_class_logits = batch_class_logits.cpu()
        # x_offset = x_offset.cpu()
        # y_offset = y_offset.cpu()
        # boxes is [x, y, w, h] where (x, y) is center

        if labels is None:  # not training
            batch_objectness_logits = torch.sigmoid(batch_objectness_logits)
            #batch_class_logits = torch.sigmoid(batch_class_logits)
            batch_class_logits = torch.nn.Softmax(dim=-1)(batch_class_logits)
            predictions = torch.cat((batch_pred_boxes, batch_objectness_logits, batch_class_logits), dim=-1)
            predictions = predictions.view(batchsize, -1, n_ch)
            return predictions

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # labels is (B, K, 5); B = batch size; K = number boxes
        # last dimension of labels is [x, y, w, h, c] with (x, y) = top left; all in pixel coordinates

        batch_gt_data = self.gt_boxes_to_feature_map(labels, grid_size)
        batch_gt_data = dtype(batch_gt_data)
        batch_gt_data = batch_gt_data.detach()

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        batch_objectness_loss = dtype(1).fill_(0)
        batch_class_loss = dtype(1).fill_(0)
        batch_xy_loss = dtype(1).fill_(0)
        batch_wh_loss = dtype(1).fill_(0)

        for b in range(batchsize):
            n = int(nlabel[b])

            # if there are no boxes in the batch element
            if n == 0:
                # ***************************************************
                # Compute the Objectness Component of the Loss
                # ***************************************************

                objectness_logits = batch_objectness_logits[b, ...]
                gt_data = batch_gt_data[b, ...]
                gt_data = gt_data.detach()
                # [batch_size, grid_size, grid_size, num_anchors, 1]
                object_mask = gt_data[..., 4:5]  # this indicates where ground truth boxes exist, since not all cells have GT objects
                object_mask = torch.autograd.Variable(object_mask.data)

                objectness_loss = bce_criterion(objectness_logits, object_mask)
                objectness_loss = torch.sum(objectness_loss)
                batch_objectness_loss = batch_objectness_loss + objectness_loss

                # class, xy, wh loss are not meaningful without ground truth boxes
                continue

            # ***************************************************
            # Compute the Objectness Component of the Loss
            # ***************************************************

            objectness_logits = batch_objectness_logits[b, ...]
            gt_data = batch_gt_data[b, ...]
            gt_data = gt_data.detach()
            # [batch_size, grid_size, grid_size, num_anchors, 1]
            object_mask = gt_data[..., 4:5]  # this indicates where ground truth boxes exist, since not all cells have GT objects
            object_mask = torch.autograd.Variable(object_mask.data)

            # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
            pred_box_xy = batch_pred_boxes[b, ..., 0:2]  # these are = [sigmoid(t_x) + c_x, sigmoid(t_y) + c_y]
            pred_box_wh = batch_pred_boxes[b, ..., 2:4]  # these are = [p_w*exp(t_w), p_h*exp(t_h)]
            pred_boxes = torch.cat((pred_box_xy, pred_box_wh), dim=-1)

            # extract out the xy and wh elements
            valid_true_box_xy = gt_data[..., 0:2]
            valid_true_box_wh = gt_data[..., 2:4]

            # for objectness the (t_x, t_y) coordinate should be (0,0) the middle of the cell after sigmoid(t_x)
            valid_true_box_xy = torch.zeros_like(valid_true_box_xy)

            # for objectness the (t_w, t_h) coordinate should be (p_w, p_w) the width and height of the anchor (prior)
            valid_true_box_wh = torch.ones_like(valid_true_box_wh)
            valid_true_box_wh = valid_true_box_wh * dtype(self.anchors)  # multiply anchors to convert to (p_w, p_h)
            valid_true_boxes = torch.cat((valid_true_box_xy, valid_true_box_wh), dim=-1)

            # remove non-valid entries
            valid_true_boxes = valid_true_boxes[object_mask[..., 0].to(torch.bool)]

            # shape: [batch_size, grid_size, grid_size, num_anchors, V]
            iou = utils.bboxes_iou_xywh_broadcast(valid_true_boxes, pred_boxes)

            # shape: [batch_size, grid_size, grid_size, num_anchors]
            best_iou, max_idx = torch.max(iou, dim=-1)

            # get_ignore_mask
            ignore_mask = torch.lt(best_iou, 0.5)  # this indicates where GT IOU is < 0.5 indicating no match
            # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
            ignore_mask = torch.unsqueeze(ignore_mask, dim=-1)

            # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
            objectness_pos_mask = object_mask  # this is the ground truth objectness which should be predicted in pred_objectness_logits as 1.0
            objectness_neg_mask = (1 - object_mask) * ignore_mask  # this is the ground truth objectness which should be predicted in pred_objectness_logits as 0.0
            objectness_valid_mask = objectness_pos_mask + objectness_neg_mask
            # elements that are not 1.0 in objectness_pos_mask and 0.0 objectness_neg_mask do not contribute to the loss
            # Yolov3 Paper page 1-2 "If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction. We use the threshold of 0.5.

            objectness_valid_mask = torch.autograd.Variable(objectness_valid_mask.data)

            objectness_loss = objectness_valid_mask * bce_criterion(objectness_logits, object_mask)
            objectness_loss = torch.sum(objectness_loss)
            batch_objectness_loss = batch_objectness_loss + objectness_loss

            # ***************************************************
            # Compute the Class Component of the Loss
            # ***************************************************
            # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
            class_logits = batch_class_logits[b, ...]
            class_targets = gt_data[..., 5:]
            class_targets = torch.autograd.Variable(class_targets.data)
            class_loss = object_mask * bce_criterion(class_logits, class_targets)
            class_loss = torch.sum(class_loss)
            batch_class_loss = batch_class_loss + class_loss

            # ***************************************************
            # Compute the XY Component of the Loss
            # ***************************************************
            # this procedure inverts the conversion in reorg_layer from (t_x, t_y, t_w, t_h) to (b_x, b_y, b_w, b_h)
            # b_x = cell_size * sigmoid(t_x) + c_x
            # b_y = cell_size * sigmoid(t_x) + c_y
            # b_w = p_w * exp(t_w)
            # b_h = p_h * exp(t_h)

            # perform conversion from (t_x, t_y) to (b_x, b_y)
            # box_xy = cell_size * sigmoid(xy) + cell_offset
            # box_xy = cell_size * sigmoid(xy) + cell_size * xy_offset
            # box_xy = cell_size * (sigmoid(xy) + xy_offset)

            # get xy coordinates in one cell from the feature_map
            # numerical range: 0 ~ 1
            # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
            true_xy = gt_data[..., 0:2]  # this is b_x, b_y (image pixel box center coordinates)

            # stride multiple of xy_offset is factored out to avoid extra computation
            # this is sigmoid(t_x)
            true_xy[..., 0] = (true_xy[..., 0] / self.stride) - x_offset[b, ...]
            true_xy[..., 1] = (true_xy[..., 1] / self.stride) - y_offset[b, ...]

            # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
            pred_xy = pred_boxes[..., 0:2]  # these are = [cell_size * sigmoid(t_x) + c_x, cell_size * sigmoid(t_y) + c_y]
            # invert box_xy = (tf.nn.sigmoid(box_xy) + xy_offset) * stride from reorg_layer
            pred_xy[..., 0] = (pred_xy[..., 0] / self.stride) - x_offset[b, ...]  # stride multiple of xy_offset is factored out to avoid extra computation
            pred_xy[..., 1] = (pred_xy[..., 1] / self.stride) - y_offset[b, ...]

            # if you want to minimize (tx, ty) directly, uncomment this code. But loss based on (tx,ty) is much more unstable, and often goes to NaN. So using this requires care.
            # true_xy needs to be in (0, 1) non-inclusive in order to be within the valid output range of sigmoid
            clip_value = 0.01
            true_xy = torch.clamp(true_xy, clip_value, (1.0 - clip_value))
            pred_xy = torch.clamp(pred_xy, clip_value, (1.0 - clip_value))

            # invert sigmoid operation
            one = dtype(1).fill_(1)
            true_xy = -torch.log(one / true_xy - one)  # invert sigmoid to get (t_x, t_y)
            pred_xy = -torch.log(one / pred_xy - one)  # invert sigmoid to get (t_x, t_y)

            # get_tw_th, numerical range: 0 ~ 1
            # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
            true_tw_th = gt_data[..., 2:4] / dtype(self.anchors)
            pred_tw_th = pred_boxes[..., 2:4] / dtype(self.anchors)  # these are = [p_w*exp(t_w), p_h*exp(t_h)]

            # for numerical stability
            # true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
            # pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)
            true_tw_th = torch.clamp(true_tw_th, 1e-9, 1e9)
            pred_tw_th = torch.clamp(pred_tw_th, 1e-9, 1e9)

            true_tw_th = torch.autograd.Variable(true_tw_th.data)
            true_xy = torch.autograd.Variable(true_xy.data)

            # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
            diff = true_xy - pred_xy
            xy_loss = torch.sum(torch.mul(diff, diff) * object_mask)
            diff = true_tw_th - pred_tw_th
            wh_loss = torch.sum(torch.mul(diff, diff) * object_mask)
            batch_xy_loss = batch_xy_loss + xy_loss
            batch_wh_loss = batch_wh_loss + wh_loss

        # normalize loss based on batch size
        batch_objectness_loss = batch_objectness_loss / batchsize
        batch_class_loss = batch_class_loss / batchsize
        batch_xy_loss = batch_xy_loss / batchsize
        batch_wh_loss = batch_wh_loss/ batchsize

        loss = batch_xy_loss + batch_wh_loss + batch_objectness_loss + batch_class_loss
        # loss is a float for some reason
        loss = loss.reshape(1, 1)
        batch_xy_loss = batch_xy_loss.detach().reshape(1, 1)
        batch_wh_loss = batch_wh_loss.detach().reshape(1, 1)
        batch_objectness_loss = batch_objectness_loss.detach().reshape(1, 1)
        batch_class_loss = batch_class_loss.detach().reshape(1, 1)
        loss = torch.cat((loss, batch_xy_loss, batch_wh_loss, batch_objectness_loss, batch_class_loss), dim=-1)

        return loss
