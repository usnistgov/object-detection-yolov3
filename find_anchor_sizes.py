import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.cluster

import bbox_utils


def find_anchors(csv_dirpath):
    csv_files = [fn for fn in os.listdir(csv_dirpath) if fn.endswith('.csv')]

    w_list = []
    h_list = []
    for fn in csv_files:
        boxes = bbox_utils.load_boxes_to_xywhc(os.path.join(csv_dirpath, fn))
        for b in range(boxes.shape[0]):
            w_list.append(boxes[b, 2])
            h_list.append(boxes[b, 3])

    X = np.hstack((np.asarray(h_list).reshape(-1, 1), np.asarray(w_list).reshape(-1, 1)))

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()

    for k in range(2, 8):
        plt.cla()
        kmeans = sklearn.cluster.KMeans(n_clusters=k)
        kmeans.fit(X)
        s = kmeans.score(X)
        print('score for {}-means = {}'.format(k, s))

        y_kmeans = kmeans.predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
        plt.xlabel('Width')
        plt.ylabel('Height')
        centers = kmeans.cluster_centers_
        print('  centers = {}'.format(centers))
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        fig.show()
        plt.savefig('scatterplot_{}_clusters.png'.format(k))
        print('View the scatterplot and determine if the clusters look appropriate. You generally want a small, medium, and large anchor for Yolo.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='find_anchor_sizes', description='Script to determine what anchors to use with yolov3.')

    parser.add_argument('--csv_dirpath', dest='csv_dirpath', type=str,
                        help='Filepath to the directory containing annotation csv files with columns [X,Y,W,H]', required=True)

    args = parser.parse_args()
    csv_dirpath = args.csv_dirpath

    find_anchors(csv_dirpath)


