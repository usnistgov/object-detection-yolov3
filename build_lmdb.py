# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import numpy as np
import os
import skimage.io
import skimage.transform
from isg_ai_pb2 import ImageYoloBoxesPair
import shutil
import lmdb
import argparse
import multiprocessing
import random

import bbox_utils

TILE_SIZE = 1024
N_WORKERS = 4


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


def process_slide_tiling(csv_filepath, img_filepath):
    boxes = bbox_utils.load_boxes_to_ltrbc(csv_filepath)
    img = skimage.io.imread(img_filepath)
    
    csv_file_name = os.path.basename(csv_filepath)
    block_key = csv_file_name.replace('.csv', '')

    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]

    img_list = list()
    box_list = list()
    key_list = list()

    delta_y = TILE_SIZE
    if height > TILE_SIZE:
        delta_y = int(TILE_SIZE - 96)
    delta_x = TILE_SIZE
    if width > TILE_SIZE:
        delta_x = int(TILE_SIZE - 96)

    for x_st in range(0, width, delta_x):
        for y_st in range(0, height, delta_y):
            x_end = x_st + TILE_SIZE
            y_end = y_st + TILE_SIZE
            if x_end > width:
                # slide box to fit within image
                dx = width - x_end
                x_st = x_st + dx
                x_end = x_end + dx
            if y_end > height:
                # slide box to fit within image
                dy = height - y_end
                y_st = y_st + dy
                y_end = y_end + dy

            if x_st < 0 or y_st < 0:
                print('Cannot use image {} as it smaller than the specified tile size'.format(img_filepath))
                # if input image is smaller than the tile size
                return img_list, box_list, key_list

            box = np.zeros((1, 4))
            box[0, 0] = x_st
            box[0, 1] = y_st
            box[0, 2] = x_end
            box[0, 3] = y_end

            # crop out the tile
            pixels = img[y_st:y_end, x_st:x_end]

            # create a full resolution mask, which will be downsampled later after random cropping
            intersection = compute_intersection(box[0, :], boxes)
            tmp_boxes = boxes[intersection > 0, :]
            new_boxes = list()
            # loop over the tile to determine which pixels belong to the foreground
            for i in range(tmp_boxes.shape[0]):
                bx_st = int(tmp_boxes[i, 0] - x_st)  # local pixels coordinate
                by_st = int(tmp_boxes[i, 1] - y_st)  # local pixels coordinate
                bx_end = int(tmp_boxes[i, 2] - x_st)  # local pixels coordinate
                by_end = int(tmp_boxes[i, 3] - y_st)  # local pixels coordinate

                bx_st = max(0, bx_st)
                by_st = max(0, by_st)
                bx_end = min(bx_end, pixels.shape[1])
                by_end = min(by_end, pixels.shape[0])

                w = bx_end - bx_st + 1
                h = by_end - by_st + 1

                # ensure minimum viable overlap with actual tile
                if w > 12 and h > 12:
                    new_boxes.append([bx_st, by_st, w, h, tmp_boxes[i, 4]])  # dim 5 is class id of the box

            if len(new_boxes) == 0:
                # if not valid boxes, create empty array
                new_boxes = np.zeros((0, 5), dtype=np.int32)
            else:
                new_boxes = np.vstack(new_boxes)

            img_list.append(pixels)
            box_list.append(new_boxes)

            if len(new_boxes) == 0:
                present_classes_str = '0'
            else:
                present_classes_str = '1'
            key_str = '{}_i{}_j{}:{}'.format(block_key, y_st, x_st, present_classes_str)
            key_list.append(key_str)

    return img_list, box_list, key_list


def generate_database(csv_files, img_files, output_folder, database_name, prefix):
    print('Generating data crops')
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    print('Opening lmdb database')
    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    print('opening multiprocessing pool with {} workers'.format(N_WORKERS))
    with multiprocessing.Pool(processes=N_WORKERS) as pool:

        stride = 100  # stride across the data, since we cannot fit all the results in memory at once
        for st_idx in range(0, len(csv_files), stride):
            print('{}/{}'.format(st_idx, len(csv_files)))
            tmp_csv_files = csv_files[st_idx:st_idx + stride]  # if numpy index goes beyond the end, its ignored
            tmp_data_files = img_files[st_idx:st_idx + stride]

            results = pool.starmap(process_slide_tiling, zip(tmp_csv_files, tmp_data_files))

            for res in results:
                img_list, box_list, key_list = res

                for j in range(len(img_list)):
                    img = img_list[j]
                    boxes = box_list[j]

                    if np.any(boxes[:,4] > 0):
                        print('what?')

                    key_str = key_list[j]

                    write_img_to_db(image_txn, img, boxes, key_str)

            image_txn.commit()
            image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()

    with open(os.path.join(output_folder, database_name, '{}_block_list.csv'.format(prefix)), 'w') as fh:
        for block_str in csv_files:
            block_str = os.path.basename(block_str)
            block_str, _ = os.path.splitext(block_str)
            fh.write('{}\n'.format(block_str))


def build_lmdb(image_folder, csv_folder, output_folder, dataset_name, train_fraction, image_format):
    if image_format.startswith('.'):
        image_format = image_format[1:]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # find the image files for which annotations exist
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    # in place shuffle
    random.shuffle(csv_files)

    # # to build mini dataset for debugging
    # csv_files = [fn for fn in csv_files if os.path.getsize(os.path.join(csv_folder, fn)) > 12]
    # csv_files = csv_files[0:100]

    img_files = [fn.replace('.csv', '.{}'.format(image_format)) for fn in csv_files]

    csv_files = [os.path.join(csv_folder, fn) for fn in csv_files]
    img_files = [os.path.join(image_folder, fn) for fn in img_files]

    idx = int(train_fraction * len(csv_files))
    train_csv_files = csv_files[0:idx]
    train_img_files = img_files[0:idx]
    test_csv_files = csv_files[idx:]
    test_img_files = img_files[idx:]

    database_name = 'train-' + dataset_name + '.lmdb'
    generate_database(train_csv_files, train_img_files, output_folder, database_name, prefix='train')

    database_name = 'test-' + dataset_name + '.lmdb'
    generate_database(test_csv_files, test_img_files, output_folder, database_name, prefix='test')


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





