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
