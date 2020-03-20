# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import skimage.io
import skimage.transform

import augment
import bbox_utils

# img_fp = 'data/images/'
# box_fp = 'data/boxes/'
img_fp = 'data-large/images/'
box_fp = 'data-large/boxes/'
fns = [fn for fn in os.listdir(img_fp) if fn.endswith('.jpg')]

tgt_h = 448
tgt_w = 576

for fn in fns:
    I = skimage.io.imread(os.path.join(img_fp, fn))
    h = I.shape[0]
    w = I.shape[1]
    scale_x = float(tgt_w) / w
    scale_y = float(tgt_h) / h

    multichannel = False
    if len(I.shape) == 3:
        multichannel = True
    I2 = skimage.transform.rescale(I, [scale_y, scale_x], preserve_range=True, multichannel=multichannel)
    I2 = I2.astype(np.uint8)

    boxes = bbox_utils.load_boxes_to_xywhc(os.path.join(box_fp, fn.replace('.jpg', '.csv')))

    crop_size = [tgt_h, tgt_w]
    reflect_x = 0
    reflect_y = 0
    boxes = augment.apply_affine_transformation_boxes(boxes, crop_size, reflect_x, reflect_y, scale_x, scale_y, 0, 0)

    if boxes is None or boxes.shape[0] == 0:
        os.remove(os.path.join(img_fp, fn))
        os.remove(os.path.join(box_fp, fn.replace('.jpg', '.csv')))
    else:
        skimage.io.imsave(os.path.join(img_fp, fn), I2)
        bbox_utils.write_boxes_from_xywhc(boxes, os.path.join(box_fp, fn.replace('.jpg', '.csv')))
