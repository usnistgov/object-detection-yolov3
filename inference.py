import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')

import argparse
import os
import numpy as np

import bbox_utils
import imagereader


def inference(image_folder, image_format, saved_model_filepath, output_folder, min_box_size):
    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if image_format.startswith('.'):
        image_format = image_format[1:]

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    # load the saved model
    model = tf.saved_model.load(saved_model_filepath)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, file_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), file_name))

        print('Loading image: {}'.format(img_filepath))
        img = imagereader.imread(img_filepath)
        img = img.astype(np.float32)

        # normalize with whole image stats
        img = imagereader.zscore_normalize(img)
        print('  img.shape={}'.format(img.shape))

        # convert HWC to CHW
        batch_data = img.transpose((2, 0, 1))
        # convert CHW to NCHW
        batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))
        batch_data = tf.convert_to_tensor(batch_data)

        boxes = model(batch_data, training=False)
        # boxes are [ltrbc]

        # strip out batch_size which is fixed at one
        boxes = np.array(boxes)
        boxes = boxes[0,]

        boxes = bbox_utils.filter_small_boxes(boxes, min_box_size)

        objectness = boxes[:, 4:5]
        class_probs = boxes[:, 5:]
        boxes = boxes[:, 0:4]

        # nms boxes per tile being passed through the network
        boxes, scores, class_label = bbox_utils.per_class_nms(boxes, objectness, class_probs)
        class_label = np.reshape(class_label, (-1, 1))
        boxes = np.concatenate((boxes, class_label), axis=-1)
        boxes = boxes.astype(np.int32)

        # write merged rois
        print('Found: {} rois'.format(boxes.shape[0]))
        output_csv_file = os.path.join(output_folder, file_name.replace(image_format, 'csv'))
        bbox_utils.write_boxes_from_ltrbc(boxes, output_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference', description='Script to detect stars with the selected model')

    parser.add_argument('--saved-model-filepath', type=str,
                        help='Filepath to the saved model to use', required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--image-folder', dest='image_folder', type=str,
                        help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--image-format', dest='image_format', type=str,
                        help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')
    parser.add_argument('--min-box-size', type=int, default=32, help='Smallest detection to consider. Default (32, 32).')

    args = parser.parse_args()

    saved_model_filepath = args.saved_model_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    image_format = args.image_format
    min_box_size = args.min_box_size

    print('Arguments:')
    print('saved_model_filepath = {}'.format(saved_model_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))
    print('min_box_size = {}'.format(min_box_size))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    str_gpu_ids = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_ids

    inference(image_folder, image_format, saved_model_filepath, output_folder, min_box_size)
