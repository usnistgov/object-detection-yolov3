import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import skimage.io
import skimage.transform


def crop_to_size(img, boxes, crop_to):
    # apply the affine transformation
    img, crop_dx, crop_dy = apply_affine_transformation(img, 0, 0, 1.0, 1.0, crop_to)

    # apply the transformation elements to the boxes
    boxes = apply_affine_transformation_boxes(boxes, crop_to, 0, 0, 1.0, 1.0, crop_dx, crop_dy)

    return img, boxes

# Boxes are [N, 5] with the columns being [x, y, w, h, class-id]
def augment_image_box_pair(img, boxes, rotation_flag=False, reflection_flag=False,
                           crop_to=None,  # size to randomly crop the image down to
                           noise_augmentation_severity=0,  # noise augmentation as a percentage of current noise
                           scale_augmentation_severity=0,  # scale augmentation as a percentage of the image size):
                           blur_augmentation_max_sigma=0,  # blur augmentation kernel maximum size):
                           box_size_augmentation_severity=0, # how much to augment the box sizes by
                           box_location_jitter_severity=0): # how much to jitter the box locations

    assert rotation_flag == False, "Rotation not implemented for image and boxes pair"
    img = np.asarray(img)

    # ensure input images are np arrays
    img = np.asarray(img, dtype=np.float32)

    debug_worst_possible_transformation = False # useful for debuging how bad images can get

    # check that the input image and mask are 2D images
    assert len(img.shape) == 2 or len(img.shape) == 3

    # convert input Nones to expected
    if noise_augmentation_severity is None:
        noise_augmentation_severity = 0
    if scale_augmentation_severity is None:
        scale_augmentation_severity = 0
    if blur_augmentation_max_sigma is None:
        blur_augmentation_max_sigma = 0
    if box_size_augmentation_severity is None:
        box_size_augmentation_severity = 0
    if box_location_jitter_severity is None:
        box_location_jitter_severity = 0

    # confirm that severity is a float between [0,1]
    assert 0 <= noise_augmentation_severity < 1
    assert 0 <= scale_augmentation_severity < 1
    assert 0 <= box_size_augmentation_severity < 1
    assert 0 <= box_location_jitter_severity < 1

    # set default augmentation parameter values (which correspond to no transformation)
    reflect_x = False
    reflect_y = False
    scale_x = 1
    scale_y = 1

    if reflection_flag:
        reflect_x = np.random.rand() > 0.5  # Bernoulli
        reflect_y = np.random.rand() > 0.5  # Bernoulli

    if scale_augmentation_severity > 0:
        max_val = 1.0 + scale_augmentation_severity
        fx = (crop_to[0] / img.shape[0])
        fy = (crop_to[1] / img.shape[1])
        min_val = max(fx, fy)
        min_val = max(min_val, 1.0 - scale_augmentation_severity)
        if debug_worst_possible_transformation:
            scale_x = min_val + (max_val - min_val) * 1
            scale_y = min_val + (max_val - min_val) * 1
        else:
            scale_x = min_val + (max_val - min_val) * np.random.rand()
            scale_y = min_val + (max_val - min_val) * np.random.rand()

    # jitter the box location and size
    boxes = augment_boxes(boxes, box_location_jitter_severity, box_size_augmentation_severity, img.shape)

    # apply the affine transformation
    img, crop_dx, crop_dy = apply_affine_transformation(img, reflect_x, reflect_y, scale_x, scale_y, crop_to)

    # apply the transformation elements to the boxes
    boxes = apply_affine_transformation_boxes(boxes, crop_to, reflect_x, reflect_y, scale_x, scale_y, crop_dx, crop_dy)

    # apply augmentations
    if noise_augmentation_severity > 0:
        sigma_max = noise_augmentation_severity * (np.max(img) - np.min(img))
        max_val = sigma_max
        min_val = -sigma_max
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        sigma_img = np.random.standard_normal(img.shape) * sigma
        img = img + sigma_img

    # apply blur augmentation
    if blur_augmentation_max_sigma > 0:
        max_val = blur_augmentation_max_sigma
        min_val = -blur_augmentation_max_sigma
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        if sigma < 0:
            sigma = 0
        if sigma > 0:
            img = scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect')

    img = np.asarray(img, dtype=np.float32)
    return img, boxes


def augment_boxes(boxes, location_jitter_percent, size_percent, img_size):
    # Boxes are [N, 5] with the columns being [x, y, w, h, class-id]

    if boxes.shape[0] == 0:
        return

    img_h = img_size[0]
    img_w = img_size[1]

    class_id = boxes[:, 4]
    x_st = boxes[:, 0]
    y_st = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    # handle position jitter
    for i in range(len(x_st)):
        jitter_sigma = location_jitter_percent * w[i]
        x_delta = int(jitter_sigma * np.random.randn())
        x_st[i] += x_delta

        jitter_sigma = location_jitter_percent * h[i]
        y_delta = int(jitter_sigma * np.random.randn())
        y_st[i] += y_delta

    # handle size jitter
    for i in range(len(x_st)):
        jitter_sigma = size_percent * w[i]
        delta = int((jitter_sigma * np.random.randn()))
        x_st[i] -= int(delta / 2)
        w[i] += delta
        # w[i] = np.max(w[i], 4) # box cannot be smaller than 4 pixels

        jitter_sigma = size_percent * h[i]
        delta = int((jitter_sigma * np.random.randn()))
        y_st[i] -= int(delta / 2)
        h[i] += delta

    x_end = x_st + w - 1
    y_end = y_st + h - 1

    # constrain to input shape
    x_st = np.maximum(x_st, 0)
    y_st = np.maximum(y_st, 0)

    x_end = np.minimum(x_end, img_w - 1)
    y_end = np.minimum(y_end, img_h - 1)

    # convert Boxes to [x, y, w, h]
    w = x_end - x_st + 1
    h = y_end - y_st + 1

    assert (np.all(h > 0) and np.all(w > 0)), 'box with zero or negative size'

    x_st = x_st.reshape(-1, 1)
    y_st = y_st.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)
    class_id = class_id.reshape(-1, 1)

    boxes = np.hstack((x_st, y_st, w, h, class_id)).astype(np.int32)
    return boxes


def apply_affine_transformation_boxes(boxes, crop_size, reflect_x, reflect_y, scale_x, scale_y, crop_dx, crop_dy):
    # Boxes are [N, 5] with the columns being [x, y, w, h, class-id]

    if boxes is None:
        return None
    if boxes.shape[0] == 0:
        return None

    # convert Boxes to [x_st, y_st, x_end, y_end]
    class_id = boxes[:, 4]
    x_st = boxes[:, 0]
    y_st = boxes[:, 1]
    x_end = boxes[:, 0] + boxes[:, 2] - 1
    y_end = boxes[:, 1] + boxes[:, 3] - 1

    x_st = x_st * scale_x - crop_dx
    x_end = x_end * scale_x - crop_dx
    y_st = y_st * scale_y - crop_dy
    y_end = y_end * scale_y - crop_dy

    h = crop_size[0]
    w = crop_size[1]

    # filter out boxes which no longer exist within the image
    idx = np.logical_or(np.logical_or(x_st >= w, y_st >= h), np.logical_or(x_end < 0, y_end < 0))
    if np.any(idx):
        idx = np.logical_not(idx)
        x_st = x_st[idx]
        y_st = y_st[idx]
        x_end = x_end[idx]
        y_end = y_end[idx]
        class_id = class_id[idx]

    # filter out boxes which have a w or h < 12, as they are not substantial enough to be detected
    delta = 12
    idx = np.logical_or(np.logical_or(x_st >= (w-delta), y_st >= (h-delta)), np.logical_or(x_end < delta, y_end < delta))
    if np.any(idx):
        idx = np.logical_not(idx)
        x_st = x_st[idx]
        y_st = y_st[idx]
        x_end = x_end[idx]
        y_end = y_end[idx]
        class_id = class_id[idx]

    # handle the case where we remove all boxes
    if len(x_st) == 0:
        return None

    # constrain to input shape
    x_st = np.maximum(x_st, 0)
    y_st = np.maximum(y_st, 0)

    x_end = np.minimum(x_end, w - 1)
    y_end = np.minimum(y_end, h - 1)

    # perform reflection
    if reflect_x:
        old_x_st = x_st
        old_x_end = x_end
        x_st = w - old_x_end
        x_end = w - old_x_st
    if reflect_y:
        old_y_st = y_st
        old_y_end = y_end
        y_st = h - old_y_end
        y_end = h - old_y_st

    # convert Boxes to [x, y, w, h]
    w = x_end - x_st + 1
    h = y_end - y_st + 1

    assert (np.all(h > 0) and np.all(w > 0)), 'box with zero or negative size'

    x_st = x_st.reshape(-1, 1)
    y_st = y_st.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)
    class_id = class_id.reshape(-1, 1)

    boxes = np.hstack((x_st, y_st, w, h, class_id)).astype(np.int32)
    return boxes


def apply_affine_transformation(I, reflect_x, reflect_y, scale_x, scale_y, crop_to):

    if len(I.shape) == 2:
        I = skimage.transform.rescale(I, scale=[scale_y, scale_x], mode='reflect', preserve_range=True)
    if len(I.shape) == 3:
        I = skimage.transform.rescale(I, scale=[scale_y, scale_x, 1], mode='reflect', preserve_range=True)

    dy = 0
    dx = 0
    delta_size_y = I.shape[0] - crop_to[0]
    delta_size_x = I.shape[1] - crop_to[1]
    if delta_size_y > 0:
        dy = int(np.random.randint(0, delta_size_y))
    if delta_size_x > 0:
        dx = int(np.random.randint(0, delta_size_x))

    I = I[dy:dy+crop_to[0], dx:dx+crop_to[1]]

    if reflect_x:
        I = np.fliplr(I)
    if reflect_y:
        I = np.flipud(I)

    return I, dx, dy
