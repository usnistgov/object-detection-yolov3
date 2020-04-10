import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# gpus_to_use must bs comma separated list of gpu ids, e.g. "1,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "0, 1" for multiple

import numpy as np
import torch.utils.data
import skimage.io
import json

import yolo_dataset
import model
import utils
import bbox_utils


def inference(image_folder, image_format, checkpoint_filepath, output_folder, min_box_size):
    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if image_format.startswith('.'):
        image_format = image_format[1:]

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    # load the saved model
    ckpt_folder, _ = os.path.split(checkpoint_filepath)
    config_fp = os.path.join(ckpt_folder, 'config.json')
    if not os.path.exists(config_fp):
        raise RuntimeError('Could not find config.json in checkpoint directory')
    with open(config_fp, 'r') as fp:
        config = json.load(fp)

    yolo_model = model.YoloV3(config)
    checkpoint = torch.load(checkpoint_filepath)
    yolo_model.load_state_dict(checkpoint['model_state_dict'])

    yolo_model = yolo_model.cuda()
    yolo_model.eval()

    print('Starting inference of file list')
    with torch.no_grad():
        for i in range(len(img_filepath_list)):
            img_filepath = img_filepath_list[i]
            _, file_name = os.path.split(img_filepath)
            print('{}/{} : {}'.format(i, len(img_filepath_list), file_name))

            print('Loading image: {}'.format(img_filepath))
            img = skimage.io.imread(img_filepath)
            orig_img = img.copy()
            img = img.astype(np.float32)

            # normalize with whole image stats
            img = yolo_dataset.YoloDataset.zscore_normalize(img)
            print('  img.shape={}'.format(img.shape))

            # convert HWC to CHW
            batch_data = yolo_dataset.YoloDataset.format_image(img)
            batch_data = np.expand_dims(batch_data, axis=0)  # add phantom batch size of 1
            batch_data = torch.from_numpy(batch_data)
            batch_data = batch_data.cuda(non_blocking=True)

            predictions = yolo_model(batch_data)
            # boxes are [x1,y1,x2,y2,c]
            predictions = utils.postprocess(predictions, yolo_model.number_classes, score_threshold=0.1, iou_threshold=0.3, min_box_size=min_box_size)
            predictions = predictions[0] # extract off batch size of 1 since postprocess returns a list of batch_size
            predictions = predictions.cpu().numpy()

            scores = predictions[:, 4]
            class_label = predictions[:, 5]
            boxes = predictions[:, 0:4]
            boxes = np.round(boxes)

            class_label = np.reshape(class_label, (-1, 1))
            boxes = np.concatenate((boxes, class_label), axis=-1)
            boxes = boxes.astype(np.int32)

            # draw boxes on the images and save
            print(boxes)
            skimage.io.imsave(os.path.join(output_folder, file_name), bbox_utils.draw_boxes(orig_img, boxes))

            # write merged rois
            print('Found: {} rois'.format(boxes.shape[0]))
            output_csv_file = os.path.join(output_folder, file_name.replace(image_format, 'csv'))
            bbox_utils.write_boxes_from_xywhc(boxes, output_csv_file)


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
    parser.add_argument('--min-box-size', type=int, default=32, help='Smallest detection to consider. Default (32, 32).')

    args = parser.parse_args()

    checkpoint_filepath = args.checkpoint_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    image_format = args.image_format
    min_box_size = args.min_box_size

    print('Arguments:')
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))
    print('min_box_size = {}'.format(min_box_size))

    inference(image_folder, image_format, checkpoint_filepath, output_folder, min_box_size)
