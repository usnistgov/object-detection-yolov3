# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise Exception('Python3 required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise Exception('Tensorflow 2.x.x required')

import multiprocessing
from multiprocessing import Process
import queue
import random
import traceback
import lmdb
import numpy as np
import augment
import os
import skimage.io
import skimage.transform
from isg_ai_pb2 import ImageYoloBoxesPair
import model





def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    std = np.std(image_data)
    mv = np.mean(image_data)
    if std <= 1.0:
        # normalize (but dont divide by zero)
        image_data = (image_data - mv)
    else:
        # z-score normalize
        image_data = (image_data - mv) / std

    return image_data


def imread(fp):
    return skimage.io.imread(fp)


def imwrite(img, fp):
    skimage.io.imsave(fp, img)


def format_image(image_data):
    # reshape into tensor (NCHW)
    image_data = np.transpose(image_data, [2, 0, 1])
    return image_data


def inverse_format_boxes(label, batch_id):
    boxes = list()

    ii,jj = np.nonzero(label[batch_id, :, :, 0, 4])
    for k in range(len(ii)):
        bb = label[batch_id, ii[k], jj[k], 0, 0:4]
        # move centered (x,y) to upper left corner
        bb[0] = bb[0] - int(bb[2] / 2)
        bb[1] = bb[1] - int(bb[3] / 2)
        boxes.append(bb)

    boxes = np.vstack(boxes)
    return boxes



class ImageReader:

    def __init__(self, img_db, anchors, use_augmentation=True, balance_classes=False, shuffle=True, num_workers=1):
        self.image_db = img_db
        self.use_augmentation = use_augmentation
        self.queue_starvation = False
        self.balance_classes = balance_classes
        self.anchors = anchors
        self.number_anchors = len(self.anchors)

        if not os.path.exists(self.image_db):
            print('Could not load database file: ')
            print(self.image_db)
            raise Exception("Missing Database")

        self.shuffle = shuffle

        random.seed()

        # get a list of keys from the lmdb
        self.keys_flat = list()
        self.keys = list()
        self.keys.append(list()) # there will always be at least one class

        self.lmdb_env = lmdb.open(self.image_db, map_size=int(2e10), readonly=True) # 20 GB
        self.lmdb_txns = list()

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
                present_classes_str = present_classes_str.split(',')
                for k in present_classes_str:
                    if len(k) == 0:
                        assert empty_images_flag
                        k = int(0)
                    else:
                        if empty_images_flag:
                            k = int(k) + 1
                        else:
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
        if self.balance_classes:
            print('Dataset Example Count by Class:')
            if empty_images_flag:
                print('  class: <empty> count: {}'.format(len(self.keys[0])))
                for i in range(1, len(self.keys)):
                    print('  class: {} count: {}'.format(i-1, len(self.keys[i])))
            else:
                for i in range(len(self.keys)):
                    print('  class: {} count: {}'.format(i, len(self.keys[i])))

        self.nb_workers = num_workers
        self.maxOutQSize = num_workers * 10
        # setup queue
        self.terminateQ = multiprocessing.Queue(maxsize=self.nb_workers)  # limit output queue size
        self.outQ = multiprocessing.Queue(maxsize=self.maxOutQSize)  # limit output queue size
        self.idQ = multiprocessing.Queue(maxsize=self.nb_workers)

        self.workers = None
        self.done = False

    def get_image_size(self):
        return self.image_size

    def get_number_classes(self):
        return self.number_classes

    def get_image_count(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat))

    def startup(self):
        self.workers = None
        self.done = False

        [self.idQ.put(i) for i in range(self.nb_workers)]
        [self.lmdb_txns.append(self.lmdb_env.begin(write=False)) for i in range(self.nb_workers)]
        # launch workers
        self.workers = [Process(target=self.__image_loader) for i in range(self.nb_workers)]

        # start workers
        for w in self.workers:
            w.start()

    def shutdown(self):
        # tell workers to shutdown
        for w in self.workers:
            self.terminateQ.put(None)

        # empty the output queue (to allow blocking workers to terminate
        nb_none_received = 0
        # empty output queue
        while nb_none_received < len(self.workers):
            try:
                while True:
                    val = self.outQ.get_nowait()
                    if val is None:
                        nb_none_received += 1
            except queue.Empty:
                pass  # do nothing

        # wait for the workers to terminate
        for w in self.workers:
            w.join()

    def __get_next_key(self):
        if self.shuffle:
            if self.balance_classes:
                # select a class to add at random from the set of classes
                label_idx = random.randint(0, len(self.keys) - 1)  # randint has inclusive endpoints
                # randomly select an example from the database of the required label
                nb_examples = len(self.keys[label_idx])

                while nb_examples == 0:
                    # select a class to add at random from the set of classes
                    label_idx = random.randint(0, len(self.keys) - 1)  # randint has inclusive endpoints
                    # randomly select an example from the database of the required label
                    nb_examples = len(self.keys[label_idx])

                img_idx = random.randint(0, nb_examples - 1)
                # lookup the database key for loading the image data
                fn = self.keys[label_idx][img_idx]
            else:
                # select a key at random from the list (does not account for class imbalance)
                fn = self.keys_flat[random.randint(0, len(self.keys_flat) - 1)]
        else:  # no shuffle
            # without shuffle you cannot balance classes
            fn = self.keys_flat[self.key_idx]
            self.key_idx += self.nb_workers
            self.key_idx = self.key_idx % len(self.keys_flat)

        return fn

    def __format_boxes(self, boxes):
        # reshape into tensor [grid_size, grid_size, num_anchors, 5 + num_classes]

        anchors = np.asarray(self.anchors, dtype=np.float32)
        num_anchors = len(anchors)

        grid_sizes = []
        a = int(self.image_size[0] / model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR)
        b = int(self.image_size[1] / model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR)
        grid_sizes.append((a, b))
        a = int(self.image_size[0] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 2))
        b = int(self.image_size[1] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 2))
        grid_sizes.append((a, b))
        a = int(self.image_size[0] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 4))
        b = int(self.image_size[1] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 4))
        grid_sizes.append((a, b))

        num_layers = len(grid_sizes)

        label = []
        for l in range(num_layers):
            # leading 1 dimension is the batch, later these will be concatentated together along that dimensions
            label.append(np.zeros((grid_sizes[l][0], grid_sizes[l][1], num_anchors, (5 + self.number_classes)), dtype=np.float32))

        # handle having no boxes for this image
        if boxes is None:
            return label
        if boxes.shape[0] == 0:
            return label

        boxes = boxes.astype(np.float32)

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
            for l in range(num_layers):
                i = np.floor(boxes[t, 1] / self.image_size[0] * grid_sizes[l][0]).astype('int32')
                j = np.floor(boxes[t, 0] / self.image_size[1] * grid_sizes[l][1]).astype('int32')

                c = boxes[t, 4].astype('int32')

                # first dimension is the batch
                label[l][i, j, n, 0:4] = boxes[t, 0:4]
                label[l][i, j, n, 4] = 1.0
                label[l][i, j, n, 5 + c] = 1.0

        return label

    def __image_loader(self):
        termimation_flag = False  # flag to control the worker shutdown
        self.key_idx = self.idQ.get()  # setup non-shuffle index to stride across flat keys properly
        try:
            datum = ImageYoloBoxesPair()  # create a datum for decoding serialized caffe_pb2 objects

            local_lmdb_txn = self.lmdb_txns[self.key_idx]

            # while the worker has not been told to terminate, loop infinitely
            while not termimation_flag:

                # poll termination queue for shutdown command
                try:
                    if self.terminateQ.get_nowait() is None:
                        termimation_flag = True
                        break
                except queue.Empty:
                    pass  # do nothing

                fn = self.__get_next_key()

                # extract the serialized image from the database
                value = local_lmdb_txn.get(fn)
                # convert from serialized representation
                datum.ParseFromString(value)

                # convert from string to numpy array
                I = np.fromstring(datum.image, dtype=datum.img_type)
                # reshape the numpy array using the dimensions recorded in the datum
                I = I.reshape((datum.img_height, datum.img_width, datum.channels))
                if np.any(I.shape != np.asarray(self.image_size)):
                    raise RuntimeError("Encountered unexpected image shape from database. Expected {}. Found {}.".format(self.image_size, I.shape))

                Boxes = np.zeros((0,5), dtype=np.int32)
                # construct mask from list of boxes
                if datum.box_count > 0:
                    # convert from string to numpy array
                    Boxes = np.fromstring(datum.boxes, dtype=datum.box_type)
                    # reshape the numpy array using the dimensions recorded in the datum
                    Boxes = Boxes.reshape(datum.box_count, 5)
                    # boxes are [x, y, width, height, class-id]

                crop_to = [self.image_size[0], self.image_size[1]]
                if self.use_augmentation:
                    # setup the image data augmentation parameters
                    rotation_flag = False
                    reflection_flag = True
                    noise_augmentation_severity = 0.03  # vary noise by x% of the dynamic range present in the image
                    scale_augmentation_severity = 0.1 #0.1 # vary size by x%
                    blur_max_sigma = 2 # pixels
                    # intensity_augmentation_severity = 0.05
                    box_size_augmentation_severity = 0.03
                    box_location_jitter_severity = 0.03

                    I = I.astype(np.float32)

                    # perform image data augmentation
                    I, Boxes = augment.augment_image_box_pair(I, Boxes,
                                                              reflection_flag=reflection_flag,
                                                              rotation_flag=rotation_flag,
                                                              crop_to=crop_to,
                                                              noise_augmentation_severity=noise_augmentation_severity,
                                                              scale_augmentation_severity=scale_augmentation_severity,
                                                              blur_augmentation_max_sigma=blur_max_sigma,
                                                              box_size_augmentation_severity=box_size_augmentation_severity,
                                                              box_location_jitter_severity=box_location_jitter_severity)

                if I.shape[0] != self.image_size[0] or I.shape[1] != self.image_size[1]:
                    I, Boxes = augment.crop_to_size(I, Boxes, crop_to)

                # format the image into a tensor
                I = format_image(I)
                I = zscore_normalize(I)

                # convert the boxes into the format expected by yolov3
                label_1, label_2, label_3 = self.__format_boxes(Boxes)

                # convert the list of images into a numpy array tensor ready for tensorflow
                I = I.astype(np.float32)
                label_1 = label_1.astype(np.float32)
                label_2 = label_2.astype(np.float32)
                label_3 = label_3.astype(np.float32)

                # add the batch in the output queue
                # this put block until there is space in the output queue (size 50)
                self.outQ.put((I, label_1, label_2, label_3))

        except Exception as e:
            print('***************** Reader Error *****************')
            print(e)
            traceback.print_exc()
            print('***************** Reader Error *****************')
        finally:
            # when the worker terminates add a none to the output so the parent gets a shutdown confirmation from each worker
            self.outQ.put(None)

    def get_example(self):
        # get a ready to train batch from the output queue and pass to to the caller
        if self.outQ.qsize() < int(0.1*self.maxOutQSize):
            if not self.queue_starvation:
                print('Input Queue Starvation !!!!')
            self.queue_starvation = True
        if self.queue_starvation and self.outQ.qsize() > int(0.5*self.maxOutQSize):
            print('Input Queue Starvation Over')
            self.queue_starvation = False
        return self.outQ.get()

    def generator(self):
        while True:
            example = self.get_example()
            if example is None:
                return
            yield example

    def get_queue_size(self):
        return self.outQ.qsize()

    def get_tf_dataset(self):
        print('Creating Dataset')
        # wrap the input queues into a Dataset
        # this sets up the imagereader class as a Python generator
        # Images come in as HWC, and are converted into CHW for network
        image_shape = tf.TensorShape((self.image_size[2], self.image_size[0], self.image_size[1]))

        grid_size0 = int(self.image_size[0] / model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR)
        grid_size1 = int(self.image_size[1] / model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR)
        label_shape_1 = tf.TensorShape((grid_size0, grid_size1, self.number_anchors, 5 + self.number_classes))
        grid_size0 = int(self.image_size[0] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 2))
        grid_size1 = int(self.image_size[1] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 2))
        label_shape_2 = tf.TensorShape((grid_size0, grid_size1, self.number_anchors, 5 + self.number_classes))
        grid_size0 = int(self.image_size[0] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 4))
        grid_size1 = int(self.image_size[1] / (model.YoloV3.NETWORK_DOWNSAMPLE_FACTOR / 4))
        label_shape_3 = tf.TensorShape((grid_size0, grid_size1, self.number_anchors, 5 + self.number_classes))

        return tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32, tf.float32, tf.float32), output_shapes=(image_shape, label_shape_1, label_shape_2, label_shape_3))
