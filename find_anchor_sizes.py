# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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


