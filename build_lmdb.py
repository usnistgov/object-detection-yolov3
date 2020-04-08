# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
from isg_ai_pb2 import ImageYoloBoxesPair
import shutil
import lmdb
import argparse
import random

import bbox_utils


def read_image(fp):
    img = skimage.io.imread(fp)
    return img


def compute_intersection(box, boxes):
    # all boxes are [left, top, right, bottom]

    intersection = 0
    if boxes.shape[0] > 0:
        # this is the iou of the box against all other boxes
        x_left = np.maximum(box[0], boxes[:, 0])
        y_top = np.maximum(box[1], boxes[:, 1])
        x_right = np.minimum(box[2], boxes[:, 2])
        y_bottom = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(y_bottom - y_top, 0) * np.maximum(x_right - x_left, 0)

    return intersection


def write_img_to_db(txn, img, boxes, key_str):
    # label 0 is background, 1 is foreground
    img = np.asarray(img, dtype=np.uint8)
    boxes = np.asarray(boxes, dtype=np.int32)

    datum = ImageYoloBoxesPair()
    if len(img.shape) == 2:
        datum.channels = 1
    elif len(img.shape) == 3:
        datum.channels = img.shape[2]
    else:
        raise RuntimeError('Invalid image dimensions: {}'.format(img.shape))
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]
    datum.image = img.tobytes()
    datum.box_count = boxes.shape[0]
    if boxes.shape[0] > 0:
        datum.boxes = boxes.tobytes()

    datum.img_type = img.dtype.str
    datum.box_type = boxes.dtype.str

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


def generate_database(csv_files, img_files, output_folder, database_name):
    print('Generating database {}'.format(database_name))
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    txn_nb = 0
    for i in range(len(img_files)):
        img_fp = img_files[i]
        csv_fp = csv_files[i]

        img = read_image(img_fp)
        boxes = bbox_utils.load_boxes_to_xywhc(csv_fp)
        present_classes = np.unique(boxes[:,4].squeeze()).astype(np.int32)
        key_str = os.path.basename(csv_fp)
        key_str, _ = os.path.splitext(key_str)
        key_str = "{}_{}".format(txn_nb, key_str)
        present_classes_list = [str(k) for k in present_classes]
        class_str = ','.join(present_classes_list)
        key_str = key_str + ':' + class_str

        txn_nb += 1
        write_img_to_db(image_txn, img, boxes, key_str)

        if txn_nb % 1000 == 0:
            image_txn.commit()
            image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()

    with open(os.path.join(output_image_lmdb_file, 'annotation_list.csv'), 'w') as fh:
        for key_str in csv_files:
            key_str = os.path.basename(key_str)
            key_str, _ = os.path.splitext(key_str)
            fh.write('{}\n'.format(key_str))


def build_lmdb(image_folder, csv_folder, output_folder, dataset_name, train_fraction, image_format):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # find the image files for which annotations exist
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    # in place shuffle
    random.shuffle(csv_files)

    img_files = [fn.replace('.csv', '.{}'.format(image_format)) for fn in csv_files]

    csv_files = [os.path.join(csv_folder, fn) for fn in csv_files]
    img_files = [os.path.join(image_folder, fn) for fn in img_files]

    idx = int(train_fraction * len(csv_files))
    train_csv_files = csv_files[0:idx]
    train_img_files = img_files[0:idx]
    test_csv_files = csv_files[idx:]
    test_img_files = img_files[idx:]

    database_name = 'train-' + dataset_name + '.lmdb'
    generate_database(train_csv_files, train_img_files, output_folder, database_name)

    database_name = 'test-' + dataset_name + '.lmdb'
    generate_database(test_csv_files, test_img_files, output_folder, database_name)


if __name__ == "__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_lmdb', description='Script which converts two folders of images and masks into a pair of lmdb databases for training.')

    parser.add_argument('--image_folder', dest='image_folder', type=str, help='filepath to the folder containing the images', required=True)
    parser.add_argument('--csv_folder', dest='csv_folder', type=str, help='filepath to the folder containing the bounding box csv files', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, help='filepath to the folder where the outputs will be placed', required=True)
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of the dataset to be used in creating the lmdb files', required=True)
    parser.add_argument('--train_fraction', dest='train_fraction', type=float, help='what fraction of the dataset to use for training (0.0, 1.0)', default=0.8)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')

    args = parser.parse_args()
    image_folder = args.image_folder
    csv_folder = args.csv_folder
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    train_fraction = args.train_fraction
    image_format = args.image_format

    build_lmdb(image_folder, csv_folder, output_folder, dataset_name, train_fraction, image_format)





