# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')

import numpy as np


class YoloV3():

    # Constants controlling the network
    BLOCK_COUNT = 8
    FILTER_COUNT = 1024
    KERNEL_SIZE = 3
    NETWORK_DOWNSAMPLE_FACTOR = 32
    WEIGHT_DECAY = 5e-4

    @staticmethod
    def conv_layer(input, fc_out, kernel, stride=1):
        output = tf.keras.layers.Conv2D(
            filters=fc_out,
            kernel_size=kernel,
            padding='same',
            activation=tf.nn.leaky_relu,
            strides=stride,
            data_format='channels_first',  # translates to NCHW
            kernel_regularizer=tf.keras.regularizers.l2(l=YoloV3.WEIGHT_DECAY))(input)
        output = tf.keras.layers.BatchNormalization(axis=1)(output)
        return output

    @staticmethod
    def feature_block(inputs, nb_reps, kernel_size, filter_count):
        layer = inputs
        for idx in range(nb_reps):
            layer = YoloV3.conv_layer(layer, fc_out=int(filter_count / 2), kernel=1)
            layer = YoloV3.conv_layer(layer, fc_out=int(filter_count), kernel=kernel_size)
            layer = tf.add(inputs, layer)
        return layer

    @staticmethod
    def yolo_block(inputs, kernel_size, filter_count):
        inputs = YoloV3.conv_layer(inputs, fc_out=int(filter_count / 2), kernel=1)
        inputs = YoloV3.conv_layer(inputs, fc_out=int(filter_count), kernel=kernel_size)
        inputs = YoloV3.conv_layer(inputs, fc_out=int(filter_count / 2), kernel=1)
        inputs = YoloV3.conv_layer(inputs, fc_out=int(filter_count), kernel=kernel_size)
        inputs = YoloV3.conv_layer(inputs, fc_out=int(filter_count / 2), kernel=1)
        route = inputs
        inputs = YoloV3.conv_layer(inputs, fc_out=int(filter_count), kernel=kernel_size)
        return route, inputs

    @staticmethod
    def broadcast_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [batch_size, grid_size, grid_size, num_anchors, 2]
        # V = number of valid ground truth boxes

        # shape: [batch_size, grid_size, grid_size, num_anchors, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [batch_size, grid_size, grid_size, num_anchors, 1, 2] & [1, V, 2] ==> [batch_size, grid_size, grid_size, num_anchors, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.0,
                                    true_box_xy - true_box_wh / 2.0)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.0,
                                    true_box_xy + true_box_wh / 2.0)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.0)

        # shape: [batch_size, grid_size, grid_size, num_anchors, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # [batch_size, grid_size, grid_size, num_anchors, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)
        return iou

    @staticmethod
    def upsample_2x(input):
        # inputs ordered as NCHW
        num_filters = input.shape.as_list()[1]
        output = tf.keras.layers.Conv2DTranspose(filters=num_filters,
                                                 kernel_size=2,
                                                 padding='same',
                                                 strides=2,
                                                 activation=None,
                                                 data_format='channels_first',  # translates to NCHW
                                                 kernel_initializer=tf.ones_initializer(),
                                                 trainable=False)(input)
        return output

    @staticmethod
    def detection_layer(input, anchors, number_classes, name):
        num_anchors = len(anchors)

        feature_map = tf.keras.layers.Conv2D(filters=int(num_anchors * (5 + number_classes)),
                                             kernel_size=1,
                                             padding='same',
                                             activation=None,
                                             strides=1,
                                             data_format='channels_first',  # translates to NCHW
                                             kernel_regularizer=tf.keras.regularizers.l2(l=YoloV3.WEIGHT_DECAY),
                                             name=name)(input)

        return feature_map

    def reorg_layer(self, feature_map):
        grid_size = feature_map.shape.as_list()[2:]

        # the stride across the full resolution image, e.g how large each cell is
        # only use spatial size of self.img_size (ignore channel depth)
        stride = tf.cast(np.asarray(self.img_size[0:2], dtype=np.float32) // np.asarray(grid_size, dtype=np.float32), tf.float32)

        # convert feature_map from NCHW to [batch_size=N, H, W, C]
        feature_map = tf.transpose(feature_map, perm=[0, 2, 3, 1])
        # separate out channel dimension into Anchors and Class Prediction
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], self.number_anchors, 5 + self.number_classes])
        feature_map = tf.cast(feature_map, tf.float32)

        box_xy, box_wh, objectness_logits, class_logits = tf.split(feature_map, [2, 2, 1, self.number_classes], axis=-1)
        # box_xy = feature_map[:, :, :, :, 0:2]
        # box_wh = feature_map[:, :, :, :, 2:4]
        # objectness_logits = feature_map[:, :, :, :, 4:5]  # ,4:5 preserves the dimension where ,4 does not
        # class_logits = feature_map[:, :, :, :, 5:]  # ,5: preserves the dimension where ,5 does not

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        xy_offset = tf.concat([x_offset, y_offset], axis=-1)
        xy_offset = tf.reshape(xy_offset, [grid_size[0], grid_size[1], 1, 2])
        xy_offset = tf.cast(xy_offset, tf.float32)

        # (c_x, c_y) the cell offsets is the integer number of the cell times the stride

        # b_x = cell_size * sigmoid(t_x) + c_x
        # b_y = cell_size * sigmoid(t_x) + c_y
        # b_w = p_w * exp(t_w)
        # b_h = p_h * exp(t_h)

        # perform conversion from (t_x, t_y) to (b_x, b_y)
        # box_xy = cell_size * sigmoid(xy) + cell_offset
        # box_xy = cell_size * sigmoid(xy) + cell_size * xy_offset
        # box_xy = cell_size * (sigmoid(xy) + xy_offset)
        box_xy = (tf.nn.sigmoid(box_xy) + xy_offset) * stride

        # last 2 dim of box_wh matches self.anchors, so broadcast happens as expected
        box_wh = tf.exp(box_wh) * self.anchors
        boxes = tf.concat([box_xy, box_wh], axis=-1)

        return xy_offset, boxes, objectness_logits, class_logits

    def convert_feature_map_to_inference_detections(self, pred_feature_map):

        results = [self.reorg_layer(feature_map) for feature_map in pred_feature_map]
        boxes_list = []
        objectness_list = []
        class_probs_list = []

        for result in results:
            xy_offset, boxes, objectness_logits, prob_logits = result
            grid_size = xy_offset.shape.as_list()[:2]

            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * self.number_anchors, 4])
            objectness_logits = tf.reshape(objectness_logits, [-1, grid_size[0] * grid_size[1] * self.number_anchors, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * self.number_anchors, self.number_classes])

            objectness = tf.sigmoid(objectness_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            objectness_list.append(objectness)
            class_probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)
        objectness = tf.concat(objectness_list, axis=1)
        class_probs = tf.concat(class_probs_list, axis=1)

        # center_x2, center_y2, width2, height2 = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        center_x = boxes[:, :, 0:1]  # ,0:1 preserves dimension where ,0 does not
        center_y = boxes[:, :, 1:2]
        width = boxes[:, :, 2:3]
        height = boxes[:, :, 3:4]

        x0 = center_x - width / 2.0
        # x0 = tf.clip_by_value(x0, 0, self.img_size[1])
        y0 = center_y - height / 2.0
        # y0 = tf.clip_by_value(y0, 0, self.img_size[0])
        x1 = center_x + width / 2.0
        # x1 = tf.clip_by_value(x1, 0, self.img_size[1])
        y1 = center_y + height / 2.0
        # y1 = tf.clip_by_value(y1, 0, self.img_size[0])

        boxes = tf.concat([x0, y0, x1, y1, objectness, class_probs], axis=-1)

        return boxes

    def compute_loss(self, feature_maps, gt_data):
        loss_xy = 0.0
        loss_wh = 0.0
        loss_conf = 0.0
        loss_class = 0.0

        for i in range(len(feature_maps)):
            fm_xy_loss, fm_wh_loss, fm_objectness_loss, fm_class_loss = self.loss_layer(feature_maps[i], gt_data[i])
            loss_xy = loss_xy + fm_xy_loss
            loss_wh = loss_wh + fm_wh_loss
            loss_conf = loss_conf + fm_objectness_loss
            loss_class = loss_class + fm_class_loss

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return total_loss, loss_xy, loss_wh, loss_conf, loss_class

    def loss_layer(self, feature_map, gt_data):
        # gt_data is shape: [batch_size, grid_size[0], grid_size[1], num_anchors, 5 + NUMBER_CLASSES]
        grid_size = feature_map.shape.as_list()[2:]

        # the stride across the full resolution image, e.g how large each cell is
        stride = tf.cast(np.asarray(self.img_size[0:2], dtype=np.float32) // np.asarray(grid_size, dtype=np.float32), tf.float32)
        batch_size = tf.cast(tf.shape(feature_map)[0], tf.float32)

        # feature_map = [N, C, H, W]
        xy_offset, pred_boxes, pred_objectness_logits, pred_class_logits = self.reorg_layer(feature_map)
        # [batch_size, grid_size, grid_size, num_anchors, 1]
        object_mask = gt_data[..., 4:5]  # this indicates where ground truth boxes exist, since not all cells have GT objects

        # ***************************************************
        # Compute the Objectness Component of the Loss
        # ***************************************************
        # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
        pred_box_xy = pred_boxes[..., 0:2]  # these are = [sigmoid(t_x) + c_x, sigmoid(t_y) + c_y]
        pred_box_wh = pred_boxes[..., 2:4]  # these are = [p_w*exp(t_w), p_h*exp(t_h)]

        gt_boxes = gt_data[..., 0:4]
        # extract out the xy and wh elements
        valid_true_box_xy = gt_boxes[..., 0:2]
        valid_true_box_wh = gt_boxes[..., 2:4]

        # for objectness the (t_x, t_y) coordinate should be (0,0) the middle of the cell after sigmoid(t_x)
        valid_true_box_xy = tf.zeros_like(valid_true_box_xy)

        # for objectness the (t_w, t_h) coordinate should be (p_w, p_w) the width and height of the anchor (prior)
        valid_true_box_wh = tf.ones_like(valid_true_box_wh)
        valid_true_box_wh = valid_true_box_wh * self.anchors  # multiply anchors to convert to (p_w, p_h)

        # remove non-valid entries
        valid_true_box_xy = tf.boolean_mask(valid_true_box_xy, tf.cast(object_mask[..., 0], 'bool'))
        valid_true_box_wh = tf.boolean_mask(valid_true_box_wh, tf.cast(object_mask[..., 0], 'bool'))

        # calc iou between detection boxes and the bounding box priors
        # shape: [batch_size, grid_size, grid_size, num_anchors, V]
        iou = YoloV3.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [batch_size, grid_size, grid_size, num_anchors]
        best_iou = tf.reduce_max(iou, axis=-1)
        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)  # this indicates where GT IOU is < 0.5 indicating no match
        # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
        objectness_pos_mask = object_mask  # this is the ground truth objectness which should be predicted in pred_objectness_logits as 1.0
        objectness_neg_mask = (1 - object_mask) * ignore_mask  # this is the ground truth objectness which should be predicted in pred_objectness_logits as 0.0
        objectness_valid_mask = objectness_pos_mask + objectness_neg_mask
        # elements that are not 1.0 in objectness_pos_mask and 0.0 objectness_neg_mask do not contribute to the loss
        # Yolov3 Paper page 1-2 "If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction. We use the threshold of 0.5.

        object_mask = tf.stop_gradient(object_mask)
        objectness_valid_mask = tf.stop_gradient(objectness_valid_mask)
        objectness_loss = objectness_valid_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_objectness_logits)
        objectness_loss = tf.reduce_sum(objectness_loss) / batch_size

        # ***************************************************
        # Compute the Class Component of the Loss
        # ***************************************************
        # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_data[..., 5:], logits=pred_class_logits)
        class_loss = tf.reduce_sum(class_loss) / batch_size

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
        true_xy = (true_xy / stride) - xy_offset

        # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
        pred_xy = pred_boxes[..., 0:2]  # these are = [cell_size * sigmoid(t_x) + c_x, cell_size * sigmoid(t_y) + c_y]
        # invert box_xy = (tf.nn.sigmoid(box_xy) + xy_offset) * stride from reorg_layer
        pred_xy = (pred_xy / stride) - xy_offset  # stride multiple of xy_offset is factored out to avoid extra computation

        # if you want to minimize (tx, ty) directly, uncomment this code. But loss based on (tx,ty) is much more unstable, and often goes to NaN. So using this requires care.
        # true_xy needs to be in (0, 1) non-inclusive in order to be within the valid output range of sigmoid
        clip_value = 0.01
        true_xy = tf.clip_by_value(true_xy, clip_value, (1.0 - clip_value))
        pred_xy = tf.clip_by_value(pred_xy, clip_value, (1.0 - clip_value))

        # invert sigmoid operation
        one = tf.constant(1.0, dtype=tf.float32)
        true_xy = -tf.math.log(one / true_xy - one)  # invert sigmoid to get (t_x, t_y)
        pred_xy = -tf.math.log(one / pred_xy - one)  # invert sigmoid to get (t_x, t_y)

        # get_tw_th, numerical range: 0 ~ 1
        # shape: [batch_size, grid_size, grid_size, num_anchors, 2]
        true_tw_th = gt_data[..., 2:4] / self.anchors
        pred_tw_th = pred_boxes[..., 2:4] / self.anchors  # these are = [p_w*exp(t_w), p_h*exp(t_h)]

        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        true_tw_th = tf.math.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.math.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        true_tw_th = tf.stop_gradient(true_tw_th)
        true_xy = tf.stop_gradient(true_xy)

        # shape: [batch_size, grid_size, grid_size, num_anchors, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask) / batch_size
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask) / batch_size

        return xy_loss, wh_loss, objectness_loss, class_loss

    def build_feature_maps(self, inputs):
        feature_map_4x_res, feature_map_2x_res, feature_map_1x_res = YoloV3.darknet53_feature_extractor(inputs)
        fm1_filter_count = int(feature_map_1x_res.shape[1])
        fm2_filter_count = int(feature_map_2x_res.shape[1])
        fm4_filter_count = int(feature_map_4x_res.shape[1])

        inputs = feature_map_1x_res
        route, inputs = YoloV3.yolo_block(inputs, YoloV3.KERNEL_SIZE, fm1_filter_count)
        feature_map_1 = YoloV3.detection_layer(inputs, self.anchors, self.number_classes, 'feature_map_1')

        inputs = YoloV3.conv_layer(route, fc_out=fm2_filter_count, kernel=1)
        inputs = YoloV3.upsample_2x(inputs)
        inputs = tf.concat([inputs, feature_map_2x_res], axis=1)

        route, inputs = YoloV3.yolo_block(inputs, YoloV3.KERNEL_SIZE, fm2_filter_count)
        feature_map_2 = YoloV3.detection_layer(inputs, self.anchors, self.number_classes, 'feature_map_2')

        inputs = YoloV3.conv_layer(route, fc_out=fm4_filter_count, kernel=1)
        inputs = YoloV3.upsample_2x(inputs)
        inputs = tf.concat([inputs, feature_map_4x_res], axis=1)

        route, inputs = YoloV3.yolo_block(inputs, YoloV3.KERNEL_SIZE, fm4_filter_count)
        feature_map_3 = YoloV3.detection_layer(inputs, self.anchors, self.number_classes, 'feature_map_3')

        return feature_map_1, feature_map_2, feature_map_3

    @staticmethod
    def darknet53_feature_extractor(inputs):
        # line 25 of https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        conv1 = YoloV3.conv_layer(inputs, fc_out=int(YoloV3.FILTER_COUNT / 32), kernel=YoloV3.KERNEL_SIZE)
        # line 35
        conv2 = YoloV3.conv_layer(conv1, fc_out=int(YoloV3.FILTER_COUNT / 16), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

        # build 1x copies of metablock 1
        mb1 = YoloV3.feature_block(conv2, nb_reps=1, kernel_size=YoloV3.KERNEL_SIZE, filter_count=int(YoloV3.FILTER_COUNT / 16))

        # line 65
        mb1_to_mb2_conv = YoloV3.conv_layer(mb1, fc_out=int(YoloV3.FILTER_COUNT / 8), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

        # build 2x copies of metablock 2
        mb2 = YoloV3.feature_block(mb1_to_mb2_conv, nb_reps=2, kernel_size=YoloV3.KERNEL_SIZE, filter_count=int(YoloV3.FILTER_COUNT / 8))

        # line 115
        mb2_to_mb3_conv = YoloV3.conv_layer(mb2, fc_out=int(YoloV3.FILTER_COUNT / 4), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

        # build Nx copies of metablock 3
        mb3 = YoloV3.feature_block(mb2_to_mb3_conv, nb_reps=YoloV3.BLOCK_COUNT, kernel_size=YoloV3.KERNEL_SIZE, filter_count=int(YoloV3.FILTER_COUNT / 4))
        route1 = mb3

        # line 286
        mb3_to_mb4_conv = YoloV3.conv_layer(mb3, fc_out=int(YoloV3.FILTER_COUNT / 2), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

        # build Nx copies of metablock 4
        mb4 = YoloV3.feature_block(mb3_to_mb4_conv, nb_reps=YoloV3.BLOCK_COUNT, kernel_size=YoloV3.KERNEL_SIZE, filter_count=int(YoloV3.FILTER_COUNT / 2))
        route2 = mb4

        # line 461
        mb4_to_mb5_conv = YoloV3.conv_layer(mb4, fc_out=YoloV3.FILTER_COUNT, kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

        # build (N/2)x copies of metablock 5
        mb5 = YoloV3.feature_block(mb4_to_mb5_conv, nb_reps=int(YoloV3.BLOCK_COUNT / 2), kernel_size=YoloV3.KERNEL_SIZE, filter_count=YoloV3.FILTER_COUNT)
        route3 = mb5
        # output_layer_name5 is tensor with shape <batch_size>, 1024, <img_size>/32, <img_size>/32
        # downsample_factor = 32

        return route1, route2, route3

    def __init__(self, global_batch_size, img_size, number_classes, anchors=None, learning_rate=1e-4):

        self.number_classes = number_classes
        self.learning_rate = learning_rate
        self.global_batch_size = global_batch_size
        self.img_size = img_size
        self.score_threshold = 0.1
        self.iou_threshold = 0.5

        if anchors is None:
            self.anchors = [(32, 32), (128, 128), (256, 256)]
        else:
            self.anchors = anchors
        self.number_anchors = len(self.anchors)

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        # self.inputs = tf.keras.Input(shape=(img_size[2], None, None))
        self.inputs = tf.keras.Input(shape=(img_size[2], img_size[0], img_size[1]))

        self.box_count_fm1 = (img_size[0] / YoloV3.NETWORK_DOWNSAMPLE_FACTOR) * (img_size[1] / YoloV3.NETWORK_DOWNSAMPLE_FACTOR)
        self.box_count_fm2 = (img_size[0] / (YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 2)) * (img_size[1] / (YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 2))
        self.box_count_fm3 = (img_size[0] / (YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 4)) * (img_size[1] / (YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 4))

        self.number_output_boxes = self.number_anchors * (self.box_count_fm1 + self.box_count_fm2 + self.box_count_fm3)
        self.output_shape = [self.number_output_boxes, 5 + self.number_classes]

        self.model, self.model_feature_maps = self.__build_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def __build_model(self):
        # Build inference Graph.
        print('    building feature map')
        feature_maps = self.build_feature_maps(self.inputs)

        print('    building reorg layer')
        boxes = self.convert_feature_map_to_inference_detections(feature_maps)
        # boxes is [batch_size, num_boxes, 6]

        yolov3_feature_maps = tf.keras.Model(self.inputs, feature_maps, name='yolov3_fm')
        yolov3 = tf.keras.Model(self.inputs, boxes, name='yolov3')
        return yolov3, yolov3_feature_maps

    def get_keras_model(self):
        return self.model

    def get_keras_feature_map_model(self):
        return self.model_feature_maps

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def train_step(self, inputs):
        (images, gt_data, loss_metric, loss_xy_metric, loss_wh_metric, loss_objectness_metric, loss_class_metric) = inputs
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            feature_maps = self.model_feature_maps(images, training=True)

            # img_size = images.get_shape().as_list()[2:]  # NCHW
            total_loss, loss_xy, loss_wh, loss_objectness, loss_class = self.compute_loss(feature_maps, gt_data)

            # average across the batch (N) with the approprite global batch size
            loss_value = tf.reduce_sum(total_loss) / self.global_batch_size

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_metric.update_state(loss_value)
        loss_xy_metric.update_state(loss_xy)
        loss_wh_metric.update_state(loss_wh)
        loss_objectness_metric.update_state(loss_objectness)
        loss_class_metric.update_state(loss_class)

        return loss_value

    @tf.function
    def dist_train_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.train_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)

        return loss_value

    def test_step(self, inputs):
        (images, gt_data, loss_metric, loss_xy_metric, loss_wh_metric, loss_objectness_metric, loss_class_metric) = inputs

        feature_maps = self.model_feature_maps(images, training=False)

        # img_size = images.get_shape().as_list()[2:]  # NCHW
        total_loss, loss_xy, loss_wh, loss_objectness, loss_class = self.compute_loss(feature_maps, gt_data)

        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(total_loss) / self.global_batch_size

        loss_metric.update_state(loss_value)
        loss_xy_metric.update_state(loss_xy)
        loss_wh_metric.update_state(loss_wh)
        loss_objectness_metric.update_state(loss_objectness)
        loss_class_metric.update_state(loss_class)

        return loss_value

    @tf.function
    def dist_test_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.test_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
        return loss_value
