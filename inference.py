# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import torch
import skimage.io
import json

import yolo_dataset
import model
import utils
import bbox_utils
import yolo_layer


EDGE_EFFECT_RANGE = 96
TILE_SIZE = 1024
SAVE_IMG_WITH_BOXES_DRAWN = False
# MIN_SCORE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3
MIN_BOX_SIZE = 12


def convert_image_to_tiles(img):
    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]

    # allocate the list of tiles and their locations in the full image
    tile_list = list()
    tile_x_location = list()
    tile_y_location = list()
    radius = EDGE_EFFECT_RANGE
    tile_size = TILE_SIZE
    assert tile_size % model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR == 0
    zone_of_responsibility_size = tile_size - 2 * radius
    assert radius % model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR == 0

    # loop over the input image cropping out (tile_size x tile_size) tiles
    for i in range(0, height, zone_of_responsibility_size):
        for j in range(0, width, zone_of_responsibility_size):
            # create a bounding box
            x_st_z = j
            y_st_z = i
            x_end_z = x_st_z + zone_of_responsibility_size
            y_end_z = y_st_z + zone_of_responsibility_size

            # pad zone of responsibility by radius
            x_st = x_st_z - radius
            y_st = y_st_z - radius
            x_end = x_end_z + radius
            y_end = y_end_z + radius

            # append this tiles locations to the list
            tile_x_location.append(x_st)
            tile_y_location.append(y_st)

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

            # add to list
            tile_list.append(tile)

    return tile_list, tile_x_location, tile_y_location


def inference_image_tiled(yolo_model, img, config, min_score_threshold):
    if type(yolo_model) is torch.nn.DataParallel:
        number_classes = yolo_model.module.number_classes
    else:
        number_classes = yolo_model.number_classes

    img_size = img.shape
    height, width, channels = img.shape
    with torch.no_grad():
        print('  converting image into tensors')
        tiles, tile_x_location, tile_y_location = convert_image_to_tiles(img)

        boxes_list = []

        # iterate over the data to be inference'd
        for i in range(len(tiles)):
            # print('tile {}/{}'.format(i, len(tiles)))
            batch_data = tiles[i].astype(np.float32)
            batch_tile_x_location = tile_x_location[i]
            batch_tile_y_location = tile_y_location[i]

            # normalize with whole image stats
            batch_data = yolo_dataset.YoloDataset.zscore_normalize(batch_data)

            # convert HWC to CHW
            batch_data = yolo_dataset.YoloDataset.format_image(batch_data)
            batch_data = np.expand_dims(batch_data, axis=0)  # add phantom batch size of 1
            batch_data = torch.from_numpy(batch_data)
            batch_data = batch_data.cuda(non_blocking=True)

            feature_maps = yolo_model(batch_data)

            predictions = list()
            for i in range(len(feature_maps)):
                feature_map = feature_maps[i].cpu().numpy()
                preds = utils.reorg_layer_np(feature_map, yolo_layer.YOLOLayer.STRIDES[i], config['number_classes'], config['anchors'])
                predictions.append(preds)
            predictions = np.concatenate(predictions, axis=1)

            predictions = utils.postprocess_numpy(predictions, number_classes, score_threshold=min_score_threshold,
                                                  iou_threshold=IOU_THRESHOLD, min_box_size=MIN_BOX_SIZE)
            predictions = predictions[0]  # extract off batch size of 1 since postprocess returns a list of batch_size
            # predictions = [x, y, w, h, score, pred_class] where (x, y) is upper left

            if predictions.shape[0] == 0:
                continue  # no boxes detected

            # round the boxes
            predictions[:, 0:4] = np.round(predictions[:, 0:4])

            # remove boxes whose centers are in the EDGE_EFFECT ghost region
            invalid_idx = np.zeros((predictions.shape[0]), dtype=np.bool)
            center_xs = predictions[:, 0] + (predictions[:, 2] / 2.0)
            center_ys = predictions[:, 1] + (predictions[:, 3] / 2.0)
            for b in range(len(center_xs)):
                cx = center_xs[b]
                cy = center_ys[b]
                # only remove boxes in the edge effect range if those boxes are not on the outside of the image
                cx_global = cx + batch_tile_x_location
                cy_global = cy + batch_tile_y_location

                # handle which boundaries
                if cy_global > EDGE_EFFECT_RANGE and cy < EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1
                if cy_global <= img_size[0] - EDGE_EFFECT_RANGE and cy >= TILE_SIZE - EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1
                if cx_global > EDGE_EFFECT_RANGE and cx < EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1
                if cx_global <= img_size[1] - EDGE_EFFECT_RANGE and cx >= TILE_SIZE - EDGE_EFFECT_RANGE:
                    invalid_idx[b] = 1

            if np.any(invalid_idx):
                predictions = predictions[invalid_idx == 0, :]

            if predictions.shape[0] > 0:
                # account for mapping the local tile pixel coordinates into global space
                predictions[:, 0] += batch_tile_x_location
                predictions[:, 1] += batch_tile_y_location

                boxes_list.append(predictions)

        if len(boxes_list) > 0:
            # merge all tiles worth of boxes into a single array
            boxes = np.concatenate(boxes_list, axis=0)
            # convert [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            # remove boxes whose centers are outside of the image area
            center_xs = (boxes[:, 0] + boxes[:, 2]) / 2.0
            center_ys = (boxes[:, 1] + boxes[:, 3]) / 2.0

            invalid_idx = np.logical_or(np.logical_or(center_xs < 0, center_xs >= width), np.logical_or(center_ys < 0, center_ys >= height))
            if np.any(invalid_idx):
                boxes = boxes[invalid_idx == 0, :]

            # constrain boxes to exist within the image domain
            boxes[boxes[:, 0] < 0, 0] = 0
            boxes[boxes[:, 0] >= width, 0] = width - 1

            boxes[boxes[:, 1] < 0, 1] = 0
            boxes[boxes[:, 1] >= height, 1] = height - 1

            boxes[boxes[:, 2] < 0, 2] = 0
            boxes[boxes[:, 2] >= width, 2] = width - 1

            boxes[boxes[:, 3] < 0, 3] = 0
            boxes[boxes[:, 3] >= height, 3] = height - 1
        else:
            boxes = np.zeros((0, 6))

    # boxes = [x1, y1, x2, y2, score, pred_class]
    return boxes


def inference_image(yolo_model, img, config, min_score_threshold):
    if type(yolo_model) is torch.nn.DataParallel:
        number_classes = yolo_model.module.number_classes
    else:
        number_classes = yolo_model.number_classes

    height, width, channels = img.shape
    with torch.no_grad():
        img = img.astype(np.float32)

        # normalize with whole image stats
        img = yolo_dataset.YoloDataset.zscore_normalize(img)

        # convert HWC to CHW
        batch_data = yolo_dataset.YoloDataset.format_image(img)
        batch_data = np.expand_dims(batch_data, axis=0)  # add phantom batch size of 1
        batch_data = torch.from_numpy(batch_data)
        batch_data = batch_data.cuda(non_blocking=True)

        feature_maps = yolo_model(batch_data)

        predictions = list()
        for i in range(len(feature_maps)):
            feature_map = feature_maps[i].cpu().numpy()
            preds = utils.reorg_layer_np(feature_map, yolo_layer.YOLOLayer.STRIDES[i], config['number_classes'], config['anchors'])
            predictions.append(preds)
        predictions = np.concatenate(predictions, axis=1)

        predictions = utils.postprocess_numpy(predictions, number_classes, score_threshold=min_score_threshold, iou_threshold=IOU_THRESHOLD, min_box_size=MIN_BOX_SIZE)
        predictions = predictions[0]  # extract off batch size of 1 since postprocess returns a list of batch_size
        # predictions = [x, y, w, h, score, pred_class] where (x, y) is upper left

        if predictions.shape[0] == 0:
            return predictions  # no boxes detected

        # round the boxes
        predictions[:, 0:4] = np.round(predictions[:, 0:4])

        # convert [x, y, w, h] to [x1, y1, x2, y2]
        predictions[:, 2] = predictions[:, 0] + predictions[:, 2]
        predictions[:, 3] = predictions[:, 1] + predictions[:, 3]

        # remove boxes whose centers are outside of the image area
        center_xs = (predictions[:, 2] + predictions[:, 0]) / 2.0
        center_ys = (predictions[:, 3] + predictions[:, 1]) / 2.0

        invalid_idx = np.logical_or(np.logical_or(center_xs < 0, center_xs >= width), np.logical_or(center_ys < 0, center_ys >= height))
        if np.any(invalid_idx):
            predictions = predictions[invalid_idx == 0, :]

        # constrain boxes to exist within the image domain
        predictions[predictions[:, 0] < 0, 0] = 0
        predictions[predictions[:, 0] >= width, 0] = width - 1

        predictions[predictions[:, 1] < 0, 1] = 0
        predictions[predictions[:, 1] >= height, 1] = height - 1

        predictions[predictions[:, 2] < 0, 2] = 0
        predictions[predictions[:, 2] >= width, 2] = width - 1

        predictions[predictions[:, 3] < 0, 3] = 0
        predictions[predictions[:, 3] >= height, 3] = height - 1

    # predictions = [x1, y1, x2, y2, score, pred_class]
    return predictions


def read_image(fp):
    from czifile import CziFile  # from https://raw.githubusercontent.com/AllenCellModeling/pytorch_fnet/master/aicsimage/io/czifile.py
    if fp.endswith('.czi'):
        with CziFile(fp) as czi:
            img = czi.asarray(resize=False)
            print("Loaded CZI image shape = ", img.shape)
            img = img.squeeze()
    else:
        img = skimage.io.imread(fp, as_gray=True)
    return img


def inference_image_folder(image_folder, image_format, checkpoint_filepath, output_folder, csv_file_list=None, min_score_threshold=0.1):
    if not os.path.exists(checkpoint_filepath):
        raise RuntimeError('Missing Checkpoint File')

    if image_format.startswith('.'):
        image_format = image_format[1:]

    img_filepath_list = [fn for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    if csv_file_list is not None:
        csv_file_list = [fn.replace('.csv','') for fn in csv_file_list]
        csv_file_list = [fn.replace('.{}'.format(image_format), '') for fn in csv_file_list]
        img_filepath_list = [fn for fn in img_filepath_list if fn.replace('.{}'.format(image_format), '') in csv_file_list]

    # prepend the folder
    img_filepath_list = [os.path.join(image_folder, fn) for fn in img_filepath_list]

    # load the saved model
    ckpt_folder, _ = os.path.split(checkpoint_filepath)
    config_fp = os.path.join(ckpt_folder, 'config.json')
    if not os.path.exists(config_fp):
        raise RuntimeError('Could not find config.json in checkpoint directory')
    with open(config_fp, 'r') as fp:
        config = json.load(fp)

    yolo_model = model.YoloV3(config)
    yolo_model = torch.nn.DataParallel(yolo_model)
    checkpoint = torch.load(checkpoint_filepath)
    yolo_model.load_state_dict(checkpoint['model_state_dict'])

    yolo_model = yolo_model.cuda()
    yolo_model.eval()

    # create output filepath
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, file_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), file_name))

        print('Loading image: {}'.format(img_filepath))
        # img = skimage.io.imread(img_filepath)
        img = read_image(img_filepath)
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        print('  img.shape={}'.format(img.shape))
        height, width, channels = img.shape

        if height > TILE_SIZE or width > TILE_SIZE:
            pass
            predictions = inference_image_tiled(yolo_model, img, config, min_score_threshold)
        else:
            predictions = inference_image(yolo_model, img, config, min_score_threshold)

        if SAVE_IMG_WITH_BOXES_DRAWN:
            # predictions = [x1, y1, x2, y2, score, pred_class]
            boxes = predictions[:, 0:4]
            boxes = np.round(boxes)

            # draw boxes on the images and save
            # convert boxes to [x, y, w, h]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            skimage.io.imsave(os.path.join(output_folder, file_name), bbox_utils.draw_boxes(img, boxes))

        # write merged rois
        print('Found: {} rois'.format(predictions.shape[0]))
        output_csv_file = os.path.join(output_folder, file_name.replace(image_format, 'csv'))
        bbox_utils.write_boxes_from_ltrbpc(predictions, output_csv_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='inference', description='Script to detect objects with the selected model')

    parser.add_argument('--checkpoint-filepath', type=str,
                        help='Filepath to the pytroch checkpoint to use', required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--image-folder', dest='image_folder', type=str,
                        help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--image-format', dest='image_format', type=str,
                        help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')
    parser.add_argument('--csv-file-list', dest='csv_file_list', type=str,help='csv file containing ', default=None)

    args = parser.parse_args()

    checkpoint_filepath = args.checkpoint_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    image_format = args.image_format
    csv_file_list = args.csv_file_list

    print('Arguments:')
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))
    if csv_file_list is not None:
        print('csv_file_list = {}'.format(csv_file_list))
        with open(csv_file_list, 'r') as fh:
            file_list = fh.readlines()
            file_list = [fn.strip() for fn in file_list]

    inference_image_folder(image_folder, image_format, checkpoint_filepath, output_folder, csv_file_list)
