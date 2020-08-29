# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import numpy as np

import torch
import torch.utils.data

import random
import lmdb
from torch.utils.data import WeightedRandomSampler

from isg_ai_pb2 import ImageYoloBoxesPair
import augment


class YoloDataset(torch.utils.data.Dataset):
    """
    Yolo v3 dataset packaged with augmentation methods
    """

    def __init__(self, lmdb_filepath, augment=True):
        self.lmdb_filepath = lmdb_filepath

        self.augment = augment

        self.__init_database()
        self.lmdb_txn = self.lmdb_env.begin(write=False)  # hopefully a shared instance across all threads

    def __init_database(self):
        random.seed()

        # get a list of keys from the lmdb
        self.keys_flat = list()
        self.keys = list()

        self.lmdb_env = lmdb.open(self.lmdb_filepath, map_size=int(3e10), readonly=True)  # 1e10 is 10 GB

        present_classes_str_flat = list()

        datum = ImageYoloBoxesPair()
        print('Initializing image database')

        with self.lmdb_env.begin(write=False) as lmdb_txn:
            # first pass to determine class count and whether or not there are images without a class
            empty_images_flag = False
            highest_class_nb = 0
            cursor = lmdb_txn.cursor().iternext(keys=True, values=False)
            for key in cursor:
                present_classes_str = key.decode('ascii').split(':')[1]
                present_classes_str = present_classes_str.split(',')
                for k in present_classes_str:
                    if len(k) == 0:
                        empty_images_flag = True
                    else:
                        highest_class_nb = max(highest_class_nb, int(k))
            for i in range(highest_class_nb):
                self.keys.append(list())
            if empty_images_flag:
                self.keys.append(list())

            # second pass to populate the database
            # get a new cursor to the start of the database
            cursor = lmdb_txn.cursor().iternext(keys=True, values=False)
            for key in cursor:
                self.keys_flat.append(key)

                present_classes_str = key.decode('ascii').split(':')[1]
                present_classes_str_split = present_classes_str.split(',')
                # TODO test this code, its an untested port of similar functionality for empty class balancing from the tensorflow imagereader.
                if len(present_classes_str_split) == 0:
                    present_classes_str = "0"
                    present_classes_str_split = present_classes_str.split(',')
                    present_classes_str_flat.append(present_classes_str)
                else:
                    if empty_images_flag:
                        present_classes_str_split = [str(int(k) + 1) for k in present_classes_str_split]
                        present_classes_str = ','.join(present_classes_str_split)
                        present_classes_str_flat.append(present_classes_str)
                    else:
                        present_classes_str_flat.append(present_classes_str)
                for k in present_classes_str_split:
                    k = int(k)
                    self.keys[k].append(key)

            # extract the serialized image from the database
            value = lmdb_txn.get(self.keys_flat[0])
            # convert from serialized representation
            datum.ParseFromString(value)
            # HWC
            self.image_size = [datum.img_height, datum.img_width, datum.channels]

        self.number_classes = len(self.keys)
        print('Found images of shape: {}'.format(self.image_size))

        print('Dataset has {} examples'.format(len(self.keys_flat)))
        print('Dataset Example Count by Class:')
        class_count = list()
        for i in range(len(self.keys)):
            if empty_images_flag:
                if i == 0:
                    print('  class: <empty> count: {}'.format(i, len(self.keys[i])))
                else:
                    print('  class: {} count: {}'.format(i-1, len(self.keys[i])))
            else:
                print('  class: {} count: {}'.format(i, len(self.keys[i])))
            class_count.append(len(self.keys[i]))

        # setup sampler for weighting samples to balance classes
        class_instance_count = 0
        for i in range(len(class_count)):
            class_instance_count += class_count[i]
        self.class_weights = np.zeros((len(class_count)))
        for i in range(len(class_count)):
            self.class_weights[i] = 1.0 - (float(class_count[i]) / float(class_instance_count))

        self.weights = np.zeros((len(self.keys_flat)))
        for i in range(len(self.keys_flat)):
            class_str = present_classes_str_flat[i]
            present_classes_str = class_str.split(',')
            sum = 0
            count = 0
            for k in present_classes_str:
                k = int(k)
                sum += self.class_weights[k]
                count += 1
            # average the weight between all classes present in that image
            self.weights[i] = sum / count

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        print('Sample weighting to balance classes: ')
        print(self.class_weights)

        samples_weight = torch.from_numpy(self.weights)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler

    def get_number_classes(self):
        return self.number_classes

    def get_image_shape(self):
        return self.image_size

    def __len__(self):
        return len(self.keys_flat)

    @staticmethod
    def format_image(x):
        # reshape into tensor (CHW)
        x = np.transpose(x, [2, 0, 1])
        return x

    @staticmethod
    def zscore_normalize(x):
        x = x.astype(np.float32)

        std = np.std(x)
        mv = np.mean(x)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            x = (x - mv)
        else:
            # z-score normalize
            x = (x - mv) / std
        return x

    def __getitem__(self, index):
        datum = ImageYoloBoxesPair()  # create a datum for decoding serialized caffe_pb2 objects
        fn = self.keys_flat[index]

        # extract the serialized image from the database
        value = self.lmdb_txn.get(fn)
        # convert from serialized representation
        datum.ParseFromString(value)

        # convert from string to numpy array
        img = np.fromstring(datum.image, dtype=datum.img_type)
        # reshape the numpy array using the dimensions recorded in the datum
        img = img.reshape((datum.img_height, datum.img_width, datum.channels))
        if np.any(img.shape != np.asarray(self.image_size)):
            raise RuntimeError("Encountered unexpected image shape from database. Expected {}. Found {}.".format(self.image_size, img.shape))

        boxes = np.zeros((0, 5), dtype=np.int32)
        # construct mask from list of boxes
        if datum.box_count > 0:
            # convert from string to numpy array
            boxes = np.fromstring(datum.boxes, dtype=datum.box_type)
            # reshape the numpy array using the dimensions recorded in the datum
            boxes = boxes.reshape(datum.box_count, 5)
            # boxes are [x, y, width, height, class-id]

        crop_to = [self.image_size[0], self.image_size[1]]
        if self.augment:
            # setup the image data augmentation parameters
            rotation_flag = False
            reflection_flag = True
            noise_augmentation_severity = 0.03  # vary noise by x% of the dynamic range present in the image
            scale_augmentation_severity = 0.1  # 0.1 # vary size by x%
            blur_max_sigma = 2  # pixels
            # intensity_augmentation_severity = 0.05
            box_size_augmentation_severity = 0.03
            box_location_jitter_severity = 0.03

            img = img.astype(np.float32)

            # perform image data augmentation
            img, boxes = augment.augment_image_box_pair(img, boxes,
                                                      reflection_flag=reflection_flag,
                                                      rotation_flag=rotation_flag,
                                                      crop_to=crop_to,
                                                      noise_augmentation_severity=noise_augmentation_severity,
                                                      scale_augmentation_severity=scale_augmentation_severity,
                                                      blur_augmentation_max_sigma=blur_max_sigma,
                                                      box_size_augmentation_severity=box_size_augmentation_severity,
                                                      box_location_jitter_severity=box_location_jitter_severity)
            if boxes is None:
                boxes = np.zeros((0, 5), dtype=np.int32)

        if img.shape[0] != self.image_size[0] or img.shape[1] != self.image_size[1]:
            img, boxes = augment.crop_to_size(img, boxes, crop_to)

        # format the image into a tensor
        img = self.format_image(img)
        img = self.zscore_normalize(img)

        # pad box list to 50 entries
        nb_boxes_to_send_to_gpu = 100
        if boxes.shape[0] > nb_boxes_to_send_to_gpu:
            raise RuntimeError('Encountered more boxes than expected')
        nb_to_pad = nb_boxes_to_send_to_gpu - boxes.shape[0]
        pad_shape = [nb_to_pad, 5]
        boxes = np.concatenate((boxes, np.zeros(pad_shape)))

        img = torch.from_numpy(img)
        boxes = torch.from_numpy(boxes)

        return img, boxes