import cv2
import numpy as np
import scipy.ndimage.interpolation
import skimage.transform
import nibabel as nib


def to_one_hot(array, depth):
    array_reshaped = np.reshape(array, -1).astype(np.uint8)
    array_one_hot = np.zeros((array_reshaped.shape[0], depth))
    array_one_hot[np.arange(array_reshaped.shape[0]), array_reshaped] = 1
    array_one_hot = np.reshape(array_one_hot, array.shape + (-1,))

    return array_one_hot


def crop_image(image, cx, cy, size, constant_values=0):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    # Copied from https://github.com/baiwenjia/ukbb_cardiac/blob/master/common/image_utils.py
    X, Y = image.shape[:2]
    rX = int(size[0] / 2)
    rY = int(size[1] / 2)
    x1, x2 = cx - rX, cx + rX
    y1, y2 = cy - rY, cy + rY
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop, x1, y1


def resize_image(image, size, interpolation_order):
    return skimage.transform.resize(image, tuple(size), order=interpolation_order, mode='constant')


def augment_image_contour_2d(whole_image, whole_contour, preserve_across_slices, max_shift=10, max_rotate=10, max_scale=0.1):
    '''
    :param whole_image: (H, W, N)
    :param whole_contour: (num_points, 2, N, C)
    :param preserve_across_slices:
    :param max_shift:
    :param max_rotate:
    :param max_scale:
    :return:
    '''
    new_whole_image = np.zeros_like(whole_image)

    if whole_contour is None:
        new_whole_contour = None
    else:
        new_whole_contour = np.zeros_like(whole_contour)

    for i in range(whole_image.shape[2]):
        image = whole_image[:, :, i]
        new_image = image

        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        if preserve_across_slices and i is not 0:
            pass
        else:
            shift_val = [np.clip(np.random.normal(), -3, 3) * max_shift,
                         np.clip(np.random.normal(), -3, 3) * max_shift]
            rotate_val = np.clip(np.random.normal(), -3, 3) * max_rotate
            scale_val = 1 + np.clip(np.random.normal(), -3, 3) * max_scale

        new_whole_image[:, :, i] = transform_data_2d(new_image, shift_val, rotate_val, scale_val, interpolation_order=1)

        if whole_contour is not None:
            row, col = image.shape

            for j in range(whole_contour.shape[3]):
                contour = whole_contour[:, :, i, j]
                new_contour = transform_coords_2d(contour, shift_val, rotate_val, scale_val, row / 2, col / 2)
                new_whole_contour[:, :, i, j] = new_contour

    return new_whole_image, new_whole_contour


def transform_data_2d(image, shift_value, rotate_value, scale_value, interpolation_order):
    # Apply the affine transformation (rotation + scale + shift) to the image
    row, col = image.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_value, 1.0 / scale_value)
    M[:, 2] += shift_value

    return scipy.ndimage.interpolation.affine_transform(image, M[:, :2], M[:, 2], order=interpolation_order)


def transform_coords_2d(coords, shift_value, rotate_value, scale_value, rotate_centre_x, rotate_centre_y):
    coords = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1)
    M = cv2.getRotationMatrix2D((rotate_centre_x, rotate_centre_y), rotate_value, 1.0 / scale_value)
    M[:, 2] += shift_value

    M = np.concatenate([M, np.array([[0, 0, 1]])], axis=0)
    M = np.linalg.inv(M)
    M = M[:2, :]

    coords_transformed = np.matmul(coords, M.transpose())

    return coords_transformed


def save_nii(image, affine, header, filename):
    if header is not None:
        nii_image = nib.Nifti1Image(image, None, header=header)
    else:
        nii_image = nib.Nifti1Image(image, affine)

    nib.save(nii_image, filename)
    return


def load_nii(nii_image):
    image = nib.load(nii_image)
    affine = image.header.get_best_affine()
    image = image.get_data()

    return image, affine
