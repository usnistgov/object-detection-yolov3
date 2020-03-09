import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
import datetime

READER_COUNT = 3  # per gpu

import numpy as np
import tensorflow as tf
if int(tf.__version__.split('.')[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import yolov3
import imagereader
import time


def train_model(batch_size, test_every_n_steps, train_database_filepath, test_database_filepath, output_folder, early_stopping_count, learning_rate, use_augmentation):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    anchors = [(32, 32), (128, 128), (256, 256)]

    training_checkpoint_filepath = None

    # uses all available devices
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # scale the batch size based on the GPU count
        global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
        # scale the number of I/O readers based on the GPU count
        reader_count = READER_COUNT * mirrored_strategy.num_replicas_in_sync

        print('Setting up test image reader')
        test_reader = imagereader.ImageReader(test_database_filepath, anchors, use_augmentation=False, shuffle=False, num_workers=reader_count)
        print('Test Reader has {} images'.format(test_reader.get_image_count()))

        print('Setting up training image reader')
        train_reader = imagereader.ImageReader(train_database_filepath, anchors, use_augmentation=use_augmentation, shuffle=True, num_workers=reader_count, balance_classes=True)
        print('Train Reader has {} images'.format(train_reader.get_image_count()))

        try:  # if any errors happen we want to catch them and shut down the multiprocess readers
            print('Starting Readers')
            train_reader.startup()
            print('  train_reader online')
            test_reader.startup()
            print('  test_reader online')

            train_dataset = train_reader.get_tf_dataset()
            train_dataset = train_dataset.batch(global_batch_size).prefetch(reader_count)
            train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

            test_dataset = test_reader.get_tf_dataset()
            test_dataset = test_dataset.batch(global_batch_size).prefetch(reader_count)
            test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

            print('Creating model')
            number_classes = train_reader.get_number_classes()
            model = yolov3.YoloV3(global_batch_size, train_reader.get_image_size(), number_classes, anchors, learning_rate)

            checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())

            # train_epoch_size = train_reader.get_image_count()/batch_size
            train_epoch_size = test_every_n_steps
            test_epoch_size = test_reader.get_image_count() / batch_size

            test_loss = list()
            # Prepare the metrics.
            train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            train_loss_xy_metric = tf.keras.metrics.Mean('train_loss_xy', dtype=tf.float32)
            train_loss_wh_metric = tf.keras.metrics.Mean('train_loss_wh', dtype=tf.float32)
            train_loss_objectness_metric = tf.keras.metrics.Mean('train_loss_obj', dtype=tf.float32)
            train_loss_class_metric = tf.keras.metrics.Mean('train_loss_class', dtype=tf.float32)

            test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
            test_loss_xy_metric = tf.keras.metrics.Mean('test_loss_xy', dtype=tf.float32)
            test_loss_wh_metric = tf.keras.metrics.Mean('test_loss_wh', dtype=tf.float32)
            test_loss_objectness_metric = tf.keras.metrics.Mean('test_loss_obj', dtype=tf.float32)
            test_loss_class_metric = tf.keras.metrics.Mean('test_loss_class', dtype=tf.float32)

            current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            train_log_dir = os.path.join(output_folder, 'tensorboard-' + current_time, 'train')
            if not os.path.exists(train_log_dir):
                os.makedirs(train_log_dir)
            test_log_dir = os.path.join(output_folder, 'tensorboard-' + current_time, 'test')
            if not os.path.exists(test_log_dir):
                os.makedirs(test_log_dir)

            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

            epoch = 0
            print('Running Network')
            while True:  # loop until early stopping
                print('---- Epoch: {} ----'.format(epoch))
                if epoch == 0:
                    cur_train_epoch_size = min(1000, train_epoch_size)
                    print('Performing Adam Optimizer learning rate warmup for {} steps'.format(cur_train_epoch_size))
                    model.set_learning_rate(learning_rate / 10)
                else:
                    cur_train_epoch_size = train_epoch_size
                    model.set_learning_rate(learning_rate)

                # Iterate over the batches of the train dataset.
                start_time = time.time()
                for step, (batch_images, label_1_batch, label_2_batch, label_3_batch) in enumerate(train_dataset):
                    if step > cur_train_epoch_size:
                        break

                    label_batch = (label_1_batch, label_2_batch, label_3_batch)
                    inputs = (batch_images, label_batch, train_loss_metric, train_loss_xy_metric, train_loss_wh_metric, train_loss_objectness_metric, train_loss_class_metric)
                    loss_value = model.dist_train_step(mirrored_strategy, inputs)
                    if np.isnan(loss_value):
                        raise RuntimeError('Training Loss went to NaN, try a lower learning rate')

                    print('Train Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, train_epoch_size, train_loss_metric.result()))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss_metric.result(), step=int(epoch * train_epoch_size + step))
                        tf.summary.scalar('loss_xy', train_loss_xy_metric.result(), step=int(epoch * train_epoch_size + step))
                        tf.summary.scalar('loss_wh', train_loss_wh_metric.result(), step=int(epoch * train_epoch_size + step))
                        tf.summary.scalar('loss_obj', train_loss_objectness_metric.result(), step=int(epoch * train_epoch_size + step))
                        tf.summary.scalar('loss_class', train_loss_class_metric.result(), step=int(epoch * train_epoch_size + step))

                    train_loss_metric.reset_states()
                    train_loss_xy_metric.reset_states()
                    train_loss_wh_metric.reset_states()
                    train_loss_objectness_metric.reset_states()
                    train_loss_class_metric.reset_states()

                # Iterate over the batches of the test dataset.
                epoch_test_loss = list()
                for step, (batch_images, label_1_batch, label_2_batch, label_3_batch) in enumerate(test_dataset):
                    if step > test_epoch_size:
                        break

                    label_batch = (label_1_batch, label_2_batch, label_3_batch)
                    inputs = (batch_images, label_batch, test_loss_metric, test_loss_xy_metric, test_loss_wh_metric, test_loss_objectness_metric, test_loss_class_metric)
                    loss_value = model.dist_test_step(mirrored_strategy, inputs)
                    if np.isnan(loss_value):
                        raise RuntimeError('Test Loss went to NaN')

                    epoch_test_loss.append(loss_value.numpy())
                    # print('Test Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, test_epoch_size, loss_value))
                test_loss.append(np.mean(epoch_test_loss))

                print('Test Epoch: {}: Loss = {}'.format(epoch, test_loss_metric.result()))
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss_metric.result(), step=int((epoch + 1) * train_epoch_size))
                    tf.summary.scalar('loss_xy', test_loss_xy_metric.result(), step=int((epoch + 1) * train_epoch_size))
                    tf.summary.scalar('loss_wh', test_loss_wh_metric.result(), step=int((epoch + 1) * train_epoch_size))
                    tf.summary.scalar('loss_obj', test_loss_objectness_metric.result(), step=int((epoch + 1) * train_epoch_size))
                    tf.summary.scalar('loss_class', test_loss_class_metric.result(), step=int((epoch + 1) * train_epoch_size))
                test_loss_metric.reset_states()
                test_loss_xy_metric.reset_states()
                test_loss_wh_metric.reset_states()
                test_loss_objectness_metric.reset_states()
                test_loss_class_metric.reset_states()

                with open(os.path.join(output_folder, 'test_loss.csv'), 'w') as csvfile:
                    for i in range(len(test_loss)):
                        csvfile.write(str(test_loss[i]))
                        csvfile.write('\n')

                print('Epoch took: {} s'.format(time.time() - start_time))

                # determine if to record a new checkpoint based on best test loss
                if (len(test_loss) - 1) == np.argmin(test_loss):
                    # save tf checkpoint
                    print('Test loss improved: {}, saving checkpoint'.format(np.min(test_loss)))
                    # checkpoint.save(os.path.join(output_folder, 'checkpoint', "ckpt")) # does not overwrite
                    training_checkpoint_filepath = checkpoint.write(os.path.join(output_folder, 'checkpoint', "ckpt"))

                # determine early stopping
                CONVERGENCE_TOLERANCE = 1e-4
                print('Best Current Epoch Selection:')
                print('Test Loss:')
                print(test_loss)
                min_test_loss = np.min(test_loss)
                error_from_best = np.abs(test_loss - min_test_loss)
                error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
                best_epoch = np.where(error_from_best == 0)[0][
                    0]  # unpack numpy array, select first time since that value has happened
                print('Best epoch: {}'.format(best_epoch))

                if len(test_loss) - best_epoch > early_stopping_count:
                    break  # break the epoch loop
                epoch = epoch + 1

        finally:  # if any errors happened during training, shut down the disk readers
            print('Shutting down train_reader')
            train_reader.shutdown()
            print('Shutting down test_reader')
            test_reader.shutdown()

    # convert training checkpoint to the saved model format
    if training_checkpoint_filepath is not None:
        print('Converting checkpoint into Saved_Model')
        # restore the checkpoint (outside of the distributed trained) and generate a saved model
        print('Model parameters:')
        print('  global_batch_size = {}'.format(global_batch_size))
        print('  img_size = {}'.format(train_reader.get_image_size()))
        print('  number_classes = {}'.format(number_classes))
        print('  anchors = {}'.format(anchors))
        print('  learning_rate = {}'.format(learning_rate))
        model = yolov3.YoloV3(global_batch_size, train_reader.get_image_size(), number_classes, anchors, learning_rate)

        checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())
        checkpoint.restore(training_checkpoint_filepath).expect_partial()
        tf.saved_model.save(model.get_keras_model(), os.path.join(output_folder, 'saved_model'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='train_yolo', description='Script which trains a yolo_v3 model')

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='training batch size', default=8)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--test_every_n_steps', dest='test_every_n_steps', type=int,
                        help='number of gradient update steps to take between test runs', default=1000)
    parser.add_argument('--train_database', dest='train_database_filepath', type=str,
                        help='lmdb database to use for (Required)', required=True)
    parser.add_argument('--test_database', dest='test_database_filepath', type=str,
                        help='lmdb database to use for testing (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_folder', type=str,
                        help='Folder where outputs will be saved (Required)', required=True)
    parser.add_argument('--early_stopping', dest='terminate_after_num_epochs_without_test_loss_improvement', type=int, help='Perform early stopping when the test loss does not improve for N epochs.', default=10)
    parser.add_argument('--use_augmentation', dest='use_augmentation', type=int,
                        help='whether to use data augmentation [0 = false, 1 = true]', default=1)



    args = parser.parse_args()

    batch_size = args.batch_size
    test_every_n_steps = args.test_every_n_steps
    train_database_filepath = args.train_database_filepath
    test_database_filepath = args.test_database_filepath
    output_folder = args.output_folder
    terminate_after_num_epochs_without_test_loss_improvement = args.terminate_after_num_epochs_without_test_loss_improvement
    learning_rate = args.learning_rate
    use_augmentation = args.use_augmentation

    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('test_every_n_steps = {}'.format(test_every_n_steps))
    print('train_database_filepath = {}'.format(train_database_filepath))
    print('test_database_filepath = {}'.format(test_database_filepath))
    print('output folder = {}'.format(output_folder))
    print('terminate_after_num_epochs_without_test_loss_improvement = {}'.format(terminate_after_num_epochs_without_test_loss_improvement))
    print('learning_rate = {}'.format(learning_rate))
    print('use_augmentation = {}'.format(use_augmentation))

    train_model(batch_size, test_every_n_steps, train_database_filepath, test_database_filepath, output_folder, terminate_after_num_epochs_without_test_loss_improvement, learning_rate, use_augmentation)
