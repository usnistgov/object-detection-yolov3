# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)


import matplotlib.pyplot as plt

import argparse
import os

import numpy as np
import skimage
import skimage.io
import skimage.measure
import csv
import bbox_utils
import shutil
import multiprocessing
import inference

GENERATE_IMAGES = False
N_WORKERS = 10
SAVE_IMAGES_FOR_THRESHOLDS = [0.1]
LOCAL_CONTEXT = 96


def draw_rois(img, rois, color=[0,0,0], offset_xy=(0, 0)):

    buff = 4

    for i in range(len(rois)):
        x_st = rois[i][0] - offset_xy[0]
        y_st = rois[i][1] - offset_xy[1]

        x_end = rois[i][2] - offset_xy[0]
        y_end = rois[i][3] - offset_xy[1]

        x_st = int(round(x_st))
        x_end = int(round(x_end))
        y_st = int(round(y_st))
        y_end = int(round(y_end))

        x_st = max(x_st - buff, 0)
        y_st = max(y_st - buff, 0)
        x_end = min(x_end + buff, img.shape[1])
        y_end = min(y_end + buff, img.shape[0])

        w = x_end - x_st + 1
        h = y_end - y_st + 1
        if w < 4 or h < 4:
            continue

        # draw a rectangle around the region of interest
        img[y_st:y_st+buff, x_st:x_end, :] = color
        img[y_end-buff:y_end, x_st:x_end, :] = color
        img[y_st:y_end, x_st:x_st+buff, :] = color
        img[y_st:y_end, x_end-buff:x_end, :] = color

    return img


def load_saved_positions(filepath, threshold=None):
    A = []

    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                vec = []
                vec.append(int(row['X']))
                vec.append(int(row['Y']))
                vec.append(int(row['W']))
                vec.append(int(row['H']))
                vec[2] = vec[0] + vec[2]  # convert from width to x_end
                vec[3] = vec[1] + vec[3]  # convert from height to y_end

                if threshold is not None:
                    thres = float(row['P'])
                    if thres >= threshold:
                        A.append(vec)
                else:
                    A.append(vec)

    # [left, top, right, bottom]
    A = np.asarray(A, dtype=np.float)
    return A


def load_saved_positions_with_prob(filepath, threshold=None):
    A = []
    p = []

    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                vec = []
                vec.append(int(row['X']))
                vec.append(int(row['Y']))
                vec.append(int(row['W']))
                vec.append(int(row['H']))
                vec[2] = vec[0] + vec[2]  # convert from width to x_end
                vec[3] = vec[1] + vec[3]  # convert from height to y_end

                thres = float(row['P'])
                if threshold is not None:
                    if thres >= threshold:
                        A.append(vec)
                        p.append(thres)
                else:
                    A.append(vec)
                    p.append(thres)

    # [left, top, right, bottom]
    A = np.asarray(A, dtype=np.float)
    return A, p


def read_image(fp):
    img = skimage.io.imread(fp, as_gray=True)
    return img


def save_conf_csv_file(csv_file, threshold, TP, TN, FP, FN):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as fh:
            fh.write('SoftmaxThreshold, TP, TN, FP, FN, Precision, Recall\n')
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    new_row = '{}, {}, {}, {}, {}, {}, {}\n'.format(threshold, TP, TN, FP, FN, precision, recall)
    with open(csv_file, 'a') as fh:
        fh.write(new_row)


# def compute_overlap_norm_by_first(box, boxes):
#     if len(box) == 0 or len(boxes) == 0:
#         return np.zeros((1,1))
#     box_area = (box[2] - box[0]) * (box[3] - box[1])
#
#     # this is the overlap of the box against all other boxes
#     x_left = np.maximum(box[0], boxes[:, 0])
#     y_top = np.maximum(box[1], boxes[:, 1])
#     x_right = np.minimum(box[2], boxes[:, 2])
#     y_bottom = np.minimum(box[3], boxes[:, 3])
#
#     # compute intersection
#     intersections = np.maximum(y_bottom - y_top, 0) * np.maximum(x_right - x_left, 0)
#     # normalize against box size
#     overlap = intersections / box_area
#     return overlap
#
#
# def compute_overlap_norm_by_second(box, boxes):
#     if len(box) == 0 or len(boxes) == 0:
#         return np.zeros((1,1))
#     boxes_area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
#
#     # this is the overlap of the box against all other boxes
#     x_left = np.maximum(box[0], boxes[:, 0])
#     y_top = np.maximum(box[1], boxes[:, 1])
#     x_right = np.minimum(box[2], boxes[:, 2])
#     y_bottom = np.minimum(box[3], boxes[:, 3])
#
#     # compute intersection
#     intersections = np.maximum(y_bottom - y_top, 0) * np.maximum(x_right - x_left, 0)
#     # normalize against box size
#     overlap = intersections / boxes_area
#     return overlap


def confusion_matrix(ref_bb, comp_bb, overlap_threshold=0.3):
    FP_list = list()
    FN_list = list()
    TP_list = list()

    for i in range(ref_bb.shape[0]):
        bb = ref_bb[i, :]

        # compute the percentage of the reference box overlapping a roi
        iou = bbox_utils.compute_iou(bb, comp_bb)
        if len((iou > overlap_threshold).nonzero()[0]):
            TP_list.append(comp_bb[np.argmax(iou), :])
        else:
            FN_list.append(bb)

    for i in range(comp_bb.shape[0]):
        bb = comp_bb[i, :]

        iou = bbox_utils.compute_iou(bb, ref_bb)
        if len((iou > overlap_threshold).nonzero()[0]) == 0:
            FP_list.append(bb)

    return TP_list, FP_list, FN_list


def compute_error_rate(img_fp, slide_name, ref_csv_filename, comp_csv_filename, overlap_threshold, prob_threshold, output_folder):

    if not os.path.exists(ref_csv_filename):
        raise IOError("Missing reference csv file")
    if not os.path.exists(comp_csv_filename):
        raise IOError("Missing comparison csv file")

    ref_bb = load_saved_positions(ref_csv_filename)

    # load positions filtering by threshold
    comp_bb = load_saved_positions(comp_csv_filename, prob_threshold)

    TP_list, FP_list, FN_list = confusion_matrix(ref_bb, comp_bb, overlap_threshold)

    TP_count = len(TP_list)
    FP_count = len(FP_list)
    FN_count = len(FN_list)

    # # load the image
    # img = skimage.io.imread(img_fp, as_gray=True)
    # # make color image
    # img = np.dstack((img, img, img))
    #
    # img = draw_rois(img, TP_list, color=[0, 0, 0])
    # img = draw_rois(img, FP_list, color=[0, 0, 255])
    # img = draw_rois(img, FN_list, color=[255, 0, 0])
    # fn = 'thres_{:0.2}_{}.jpg'.format(prob_threshold, slide_name)
    # skimage.io.imsave(os.path.join(output_folder, fn), img)

    if GENERATE_IMAGES and prob_threshold in SAVE_IMAGES_FOR_THRESHOLDS:
        if FN_count > 0:
            # load the image
            img = skimage.io.imread(img_fp, as_gray=True)
            # make color image
            img = np.dstack((img, img, img))

            # img = draw_rois(img, TP_list, color=[0, 0, 0])
            # img = draw_rois(img, FP_list, color=[0, 0, 255])
            # img = draw_rois(img, FN_list, color=[255, 0, 0])
            # fn = 'FN{}_{}.jpg'.format(FN_count, slide_name)
            # skimage.io.imsave(os.path.join(output_folder, fn), img)

            fn_nb = 0
            for FN in FN_list:
                x_st = FN[0] - LOCAL_CONTEXT
                y_st = FN[1] - LOCAL_CONTEXT
                x_end = FN[2] + LOCAL_CONTEXT
                y_end = FN[3] + LOCAL_CONTEXT

                x_st = int(max(0, x_st))
                y_st = int(max(0, y_st))
                x_end = int(min(img.shape[1], x_end))
                y_end = int(min(img.shape[0], y_end))

                sub_img = img[y_st:y_end, x_st:x_end]
                sub_img = draw_rois(sub_img, TP_list, color=[0, 0, 0], offset_xy=(x_st, y_st))
                sub_img = draw_rois(sub_img, FP_list, color=[0, 0, 255], offset_xy=(x_st, y_st))
                sub_img = draw_rois(sub_img, FN_list, color=[255, 0, 0], offset_xy=(x_st, y_st))

                fn = 'FN{:04d}_{}.jpg'.format(fn_nb, slide_name)
                skimage.io.imsave(os.path.join(output_folder, fn), sub_img)
                fn_nb = fn_nb + 1

    print('Slide {}'.format(slide_name))
    print('TP Count: {}'.format(TP_count))
    print('FP Count: {}'.format(FP_count))
    print('FN Count: {}'.format(FN_count))
    print('*********************')

    return TP_count, FP_count, FN_count


def generate(slide_filename, annotation_folder, inference_folder, image_folder, output_folder, overlap_threshold, prob_threshold):

    slide_name = slide_filename.replace('.csv', '')
    print(slide_name)

    inference_csv_file = os.path.join(inference_folder, '{}.csv'.format(slide_name))

    annotation_csv_file = os.path.join(annotation_folder, '{}.csv'.format(slide_name))

    inference_image_file = os.path.join(image_folder, '{}.jpg'.format(slide_name))

    if os.path.exists(annotation_csv_file):
        TP, FP, FN = compute_error_rate(inference_image_file, slide_name, annotation_csv_file, inference_csv_file, overlap_threshold, prob_threshold, output_folder)
    else:
        TP = 0
        FP = 0
        FN = 0
    return TP, FP, FN


def main():
    parser = argparse.ArgumentParser(prog='compute_confusion_matrix', description='Script to take generate the confusion matrix information between an inference results and reference annotations')

    parser.add_argument('--model_folder', dest='model_folder', type=str, required=True)
    parser.add_argument('--annotation_folder', dest='annotation_folder', type=str, required=True)
    parser.add_argument('--image_folder', dest='image_folder', type=str, required=True)
    parser.add_argument('--overlap_threshold', dest='overlap_threshold', type=float, default=0.3)
    parser.add_argument('--prob_threshold', dest='prob_threshold', type=float, default=None)
    parser.add_argument('--slide_csv_file_list', dest='slide_csv_file_list', type=str, default=None)

    args = parser.parse_args()

    model_folder = args.model_folder
    annotation_folder = args.annotation_folder
    image_folder = args.image_folder
    overlap_threshold = args.overlap_threshold
    prob_threshold = args.prob_threshold
    slide_csv_file_list = args.slide_csv_file_list

    print('Arguments:')
    print('model_folder = {}'.format(model_folder))
    print('annotation_folder = {}'.format(annotation_folder))
    print('image_folder = {}'.format(image_folder))
    print('overlap_threshold = {}'.format(overlap_threshold))
    print('prob_threshold = {}'.format(prob_threshold))
    if slide_csv_file_list is not None:
        print('slide_csv_file_list = {}'.format(slide_csv_file_list))

    checkpoint_filepath = os.path.join(model_folder, 'checkpoint','yolov3.ckpt')
    inference_folder = os.path.join(model_folder, 'inference')
    confusion_folder = os.path.join(model_folder, 'confusion')
    if os.path.exists(inference_folder):
        shutil.rmtree(inference_folder)
    os.mkdir(inference_folder)
    if os.path.exists(confusion_folder):
        shutil.rmtree(confusion_folder)
    os.mkdir(confusion_folder)

    if prob_threshold is not None:

        if slide_csv_file_list is None:
            slide_filename_list = os.listdir(annotation_folder)
            slide_filename_list = [fn for fn in slide_filename_list if fn.endswith('.csv')]
        else:
            with open(slide_csv_file_list, 'r') as fh:
                slide_filename_list = fh.readlines()
                slide_filename_list = [fn.strip() for fn in slide_filename_list]

        inference.inference_image_folder(image_folder, 'jpg', checkpoint_filepath, inference_folder, slide_filename_list, prob_threshold)

        TP_count = 0
        FP_count = 0
        FN_count = 0

        with multiprocessing.Pool(processes=N_WORKERS) as pool:
            tmp_list = list()
            for fn in slide_filename_list:
                tmp_list.append((fn, annotation_folder, inference_folder, image_folder, confusion_folder, overlap_threshold, prob_threshold))

            results = pool.starmap(generate, tmp_list)

            for res in results:
                TP, FP, FN = res
                TP_count += TP
                FP_count += FP
                FN_count += FN

        print('saving global confusion')
        save_conf_csv_file(os.path.join(confusion_folder, 'confusion.csv'), prob_threshold, TP_count, np.NaN, FP_count, FN_count)

    else:
        # perform sweep of softmax thresholds
        thresholds = list(range(1, 100))
        thresholds = [th / 100.0 for th in thresholds]

        if slide_csv_file_list is None:
            slide_filename_list = os.listdir(annotation_folder)
            slide_filename_list = [fn for fn in slide_filename_list if fn.endswith('.csv')]
        else:
            with open(slide_csv_file_list, 'r') as fh:
                slide_filename_list = fh.readlines()
                slide_filename_list = [fn.strip() for fn in slide_filename_list]

        prob_threshold = thresholds[0]
        inference.inference_image_folder(image_folder, 'jpg', checkpoint_filepath, inference_folder, slide_filename_list, prob_threshold)

        TP = np.zeros((len(thresholds)))
        FP = np.zeros((len(thresholds)))
        FN = np.zeros((len(thresholds)))
        for th_idx in range(len(thresholds)):
            prob_threshold = thresholds[th_idx]

            TP_count = 0
            FP_count = 0
            FN_count = 0

            cur_output_folder = os.path.join(confusion_folder, 'threshold_{:.2}'.format(prob_threshold))
            if prob_threshold in SAVE_IMAGES_FOR_THRESHOLDS:
                if not os.path.exists(cur_output_folder):
                    os.mkdir(cur_output_folder)

            with multiprocessing.Pool(processes=N_WORKERS) as pool:
                tmp_list = list()
                for fn in slide_filename_list:
                    tmp_list.append((fn, annotation_folder, inference_folder, image_folder, cur_output_folder, overlap_threshold, prob_threshold))

                results = pool.starmap(generate, tmp_list)

                for res in results:
                    TP_per_img, FP_per_img, FN_per_img = res
                    TP_count += TP_per_img
                    FP_count += FP_per_img
                    FN_count += FN_per_img

            TP[th_idx] = TP_count
            FP[th_idx] = FP_count
            FN[th_idx] = FN_count

            print('saving global confusion')
            save_conf_csv_file(os.path.join(confusion_folder, 'roi_threshold_confusion.csv'), prob_threshold, TP_count, np.NaN, FP_count, FN_count)

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        dot_size = 10
        fig = plt.figure(figsize=(16, 9), dpi=200)
        ax = plt.gca()
        ax.plot(thresholds, precision, 'o-', color='b', markersize=dot_size)
        ax.plot(thresholds, recall, 'o-', color='r', markersize=dot_size)
        plt.xlabel('Probability Threshold')
        fig.savefig(os.path.join(confusion_folder, 'Precision_and_Recall.png'))

        plt.clf()
        ax = plt.gca()
        ax.plot(precision, recall, 'o-', color='b', markersize=dot_size)
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        fig.savefig(os.path.join(confusion_folder, 'Precision_vs_Recall.png'))

        plt.clf()
        ax = plt.gca()
        ax.plot(thresholds, TP, 'o-', color='b', markersize=dot_size)
        plt.ylabel('True Positive Count')
        plt.xlabel('Probability Threshold')
        fig.savefig(os.path.join(confusion_folder, 'TP.png'))

        plt.clf()
        ax = plt.gca()
        ax.plot(thresholds, FP, 'o-', color='b', markersize=dot_size)
        plt.ylabel('False Positive Count')
        plt.xlabel('Probability Threshold')
        fig.savefig(os.path.join(confusion_folder, 'FP.png'))

        plt.clf()
        ax = plt.gca()
        ax.plot(thresholds, FN, 'o-', color='b', markersize=dot_size)
        plt.ylabel('False Negative Count')
        plt.xlabel('Probability Threshold')
        fig.savefig(os.path.join(confusion_folder, 'FN.png'))

        plt.close(fig)


if __name__ == "__main__":
    main()