import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise Exception('Tensorflow 2.x.x required')

import argparse
import yolov3


def main(checkpoint_filepath, saved_model_filepath):

    # These parameters need to match what was used to train the model
    crop_img_size = [384, 512, 3]
    number_classes = 2
    anchors = [(32, 32), (128, 128), (256, 256)]

    # restore the checkpoint and generate a saved model
    model = yolov3.YoloV3(1, crop_img_size, number_classes, anchors, 1e-4)
    checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())
    checkpoint.restore(checkpoint_filepath).expect_partial()

    tf.saved_model.save(model.get_keras_model(), saved_model_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference', description='Script to detect stars with the selected model')

    parser.add_argument('--checkpoint-filepath', dest='checkpoint_filepath', type=str,
                        help='Checkpoint filepath to the  model to use', required=True)
    parser.add_argument('--saved-model-filepath', type=str, required=True)

    args = parser.parse_args()

    checkpoint_filepath = args.checkpoint_filepath
    saved_model_filepath = args.saved_model_filepath

    print('Arguments:')
    print('checkpoint_filepath = {}'.format(checkpoint_filepath))
    print('saved_model_filepath = {}'.format(saved_model_filepath))

    main(checkpoint_filepath, saved_model_filepath)