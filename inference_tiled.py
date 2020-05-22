import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise Exception('Tensorflow 2.x.x required')

import argparse
import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# gpus_to_use must bs comma separated list of gpu ids, e.g. "1,3,4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1" for multiple

import numpy as np
import skimage.io

import model
import bbox_utils
import imagereader

BATCH_SIZE = 1
EDGE_EFFECT_RANGE = 96


def convert_image_to_tiles(img, tile_size):
    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]

    # allocate the list of tiles and their locations in the full image
    tile_list = list()
    tile_x_location = list()
    tile_y_location = list()
    radius = [EDGE_EFFECT_RANGE, EDGE_EFFECT_RANGE]
    assert tile_size[0] % model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR == 0
    assert tile_size[1] % model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR == 0
    if tile_size[0] >= height:
        radius[0] = 0
    if tile_size[1] >= width:
        radius[1] = 0
    zone_of_responsibility_size = [tile_size[0] - 2 * radius[0], tile_size[1] - 2 * radius[1]]

    assert radius[0] % model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR == 0
    assert radius[1] % model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR == 0

    # loop over the input image cropping out (tile_size x tile_size) tiles
    for i in range(0, height, zone_of_responsibility_size[0]):
        for j in range(0, width, zone_of_responsibility_size[1]):
            # create a bounding box
            x_st_z = j
            y_st_z = i
            x_end_z = x_st_z + zone_of_responsibility_size[1]
            y_end_z = y_st_z + zone_of_responsibility_size[0]

            # pad zone of responsibility by radius
            x_st = x_st_z - radius[1]
            y_st = y_st_z - radius[0]
            x_end = x_end_z + radius[1]
            y_end = y_end_z + radius[0]

            pre_pad_x = 0
            if x_st < 0:
                pre_pad_x = -x_st
                x_st = 0

            pre_pad_y = 0
            if y_st < 0:
                pre_pad_y = -y_st
                y_st = 0

            post_pad_x = 0
            if x_end > width:
                post_pad_x = x_end - width
                x_end = width

            post_pad_y = 0
            if y_end > height:
                post_pad_y = y_end - height
                y_end = height

            # crop out the tile
            tile = img[y_st:y_end, x_st:x_end]

            if pre_pad_x > 0 or post_pad_x > 0 or pre_pad_y > 0 or post_pad_y > 0:
                # ensure its correct size (if tile exists at the edge of the image
                tile = np.pad(tile, pad_width=((pre_pad_y, post_pad_y), (pre_pad_x, post_pad_x), (0, 0)), mode='reflect')

            # append this tiles locations to the list
            tile_x_location.append(x_st)
            tile_y_location.append(y_st)

            # add to list
            tile_list.append(tile)

    return tile_list, tile_x_location, tile_y_location


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


def per_class_nms(boxes, objectness, class_probs, iou_threshold=0.3, score_threshold=0.1):
    # all boxes belong to the same image

    num_classes = class_probs.shape[1]
    scores = class_probs * objectness  # create blend of objectness and probs
    scores = np.sqrt(scores)  # undo the probability squaring above

    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(num_classes):
        indices = np.where(scores[:, i] >= score_threshold)

        filter_boxes = boxes[indices]
        filter_scores = scores[:, i][indices]
        if len(filter_boxes) == 0:
            continue

        # do non_max_suppression on the cpu
        indices = single_class_nms(filter_boxes, filter_scores, iou_threshold=iou_threshold)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32') * i)

    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def filter_small_boxes(boxes, min_roi_size):
    # filter out any boxes which have a width or height < 32 pixels
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    idx = np.logical_and(w > min_roi_size, h > min_roi_size)
    boxes = boxes[idx, :]
    return boxes


def inference_image_tiled(yolo_model, img, tile_size, min_roi_size):

    print('reading image')
    img_size = img.shape

    # convert the image into tiles to be fed to the network
    print('converting image into tensors')
    tile_tensor, tile_x_location, tile_y_location = convert_image_to_tiles(img, tile_size)

    boxes_list = []
    scores_list = []
    class_label_list = []

    # iterate over the data to be inference'd
    for i in range(len(tile_tensor)):
        print('tile {}/{}'.format(i, len(tile_tensor)))
        # reshape to 1d array for TRT
        batch_data = tile_tensor[i].astype(np.float32)

        # normalize with whole image stats
        batch_data = imagereader.zscore_normalize(batch_data)
        batch_tile_x_location = tile_x_location[i]
        batch_tile_y_location = tile_y_location[i]

        # convert HWC to CHW
        batch_data = batch_data.transpose((2, 0, 1))
        # convert CHW to NCHW
        batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))
        batch_data = tf.convert_to_tensor(batch_data)

        boxes = yolo_model(batch_data, training=False)

        # strip out batch_size which is fixed at one
        boxes = np.array(boxes)
        boxes = boxes[0,]

        boxes = filter_small_boxes(boxes, min_roi_size)

        objectness = boxes[:, 4:5]
        class_probs = boxes[:, 5:]
        boxes = boxes[:, 0:4]

        # nms boxes per tile being passed through the network
        boxes, scores, class_label = per_class_nms(boxes, objectness, class_probs)

        if boxes is not None:
            # reshape to match boxes[N, 1]
            scores = scores.reshape((-1, 1))
            class_label = class_label.reshape((-1, 1))

            # TODO: figure out how to handle edge effects when tiles might be smaller than the tile size. This scheme below no longer works given the new tiler.
            # remove boxes whose centers are in the EDGE_EFFECT ghost region
            invalid_idx = np.zeros((boxes.shape[0]), dtype=np.bool)
            center_xs = (boxes[:, 2] + boxes[:, 0]) / 2.0
            center_ys = (boxes[:, 3] + boxes[:, 1]) / 2.0
            for b in range(len(center_xs)):
                cx = center_xs[b]
                cy = center_ys[b]
                # handle which boundaries
                if cy < EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1
                if cy >= tile_size[0] - EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1
                if cx < EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1
                if cx >= tile_size[1] - EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1

            if np.any(invalid_idx):
                boxes = boxes[invalid_idx == 0, :]
                scores = scores[invalid_idx == 0]
                class_label = class_label[invalid_idx == 0]

            if boxes.shape[0] > 0:
                # account for mapping the local tile pixel coordinates into global space
                boxes[:, 0] += batch_tile_x_location
                boxes[:, 2] += batch_tile_x_location
                boxes[:, 1] += batch_tile_y_location
                boxes[:, 3] += batch_tile_y_location

                boxes_list.append(boxes)
                scores_list.append(scores)
                class_label_list.append(class_label)

    if len(boxes_list) > 0:
        # merge all tiles worth of boxes into a single array
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        class_label = np.concatenate(class_label_list, axis=0)

        boxes = np.round(boxes).astype(np.int32)

        # remove boxes whose centers are outside of the image area
        center_xs = (boxes[:, 2] + boxes[:, 0]) / 2.0
        center_ys = (boxes[:, 3] + boxes[:, 1]) / 2.0

        invalid_idx = np.logical_or(np.logical_or(center_xs < 0, center_xs >= img_size[1]), np.logical_or(center_ys < 0, center_ys >= img_size[0]))
        if np.any(invalid_idx):
            boxes = boxes[invalid_idx == 0, :]
            scores = scores[invalid_idx == 0]
            class_label = class_label[invalid_idx == 0]

        # constrain boxes to exist within the image domain
        boxes[boxes[:, 0] < 0, 0] = 0
        boxes[boxes[:, 0] >= img_size[1], 0] = img_size[1] - 1

        boxes[boxes[:, 1] < 0, 1] = 0
        boxes[boxes[:, 1] >= img_size[0], 1] = img_size[0] - 1

        boxes[boxes[:, 2] < 0, 2] = 0
        boxes[boxes[:, 2] >= img_size[1], 2] = img_size[1] - 1

        boxes[boxes[:, 3] < 0, 3] = 0
        boxes[boxes[:, 3] >= img_size[0], 3] = img_size[0] - 1
    else:
        boxes = np.zeros((0, 4))
        scores = np.zeros((0, 1))
        class_label = np.zeros((0, 1))

    # write merged rois
    print('Found: {} rois'.format(boxes.shape[0]))
    predictions = np.concatenate((boxes,scores, class_label), axis=-1)
    return predictions


def inference_image_folder(image_folder, image_format, saved_model_filepath, output_folder, tile_size, min_roi_size):
    if not os.path.exists(saved_model_filepath):
        raise RuntimeError('Missing saved_model_filepath File')

    if image_format.startswith('.'):
        image_format = image_format[1:]

    img_filepath_list = [fn for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    # prepend the folder
    img_filepath_list = [os.path.join(image_folder, fn) for fn in img_filepath_list]

    yolo_model = tf.saved_model.load(saved_model_filepath)

    # create output filepath
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, file_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), file_name))

        print('Loading image: {}'.format(img_filepath))
        img = skimage.io.imread(img_filepath)
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        print('  img.shape={}'.format(img.shape))

        predictions = inference_image_tiled(yolo_model, img, tile_size, min_roi_size)

        # write merged rois
        print('Found: {} rois'.format(predictions.shape[0]))
        output_csv_file = os.path.join(output_folder, file_name.replace(image_format, 'csv'))
        bbox_utils.write_boxes_from_ltrbpc(predictions, output_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference', description='Script to detect stars with the selected model')

    parser.add_argument('--saved-model-filepath', type=str,
                        help='Filepath to the saved model to use', required=True)
    parser.add_argument('--image-folder', type=str, help='Filepath to the folder of images to inference', required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--tile-height', type=int, default=512)
    parser.add_argument('--tile-width', type=int, default=512)
    parser.add_argument('--min-box-size', type=int, default=32)
    parser.add_argument('--image-format', dest='image_format', type=str,
                        help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')

    args = parser.parse_args()

    saved_model_filepath = args.saved_model_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    tile_size = [args.tile_height, args.tile_width]
    min_box_size = args.min_box_size
    image_format = args.image_format

    print('Arguments:')
    print('saved_model_filepath = {}'.format(saved_model_filepath))
    print('image_filepath = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('tile_size = {}'.format(tile_size))
    print('min_box_size = {}'.format(min_box_size))
    print('image_format = {}'.format(image_format))

    inference_image_folder(image_folder, image_format, saved_model_filepath, output_folder, tile_size, min_box_size)
