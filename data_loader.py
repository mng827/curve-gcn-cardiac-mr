from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.transforms as transforms
import torchvision
import torch

import numpy as np
import nibabel as nib
import json_tricks
import os
import image_utils
import csv
from scipy import interpolate
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_pytorch(config):
    if config.dataset == 'ukbb':
        trainset = VolumetricImageDataset(data_root_dir=config.train_data_path,
                                          data=get_image_list(config.train_data_file),
                                          image_size=config.image_size,
                                          network_type=config.network_type,
                                          num_images_limit=config.num_images_limit,
                                          augment=config.data_aug,
                                          shift=config.data_aug_shift,
                                          rotate=config.data_aug_rotate,
                                          scale=config.data_aug_scale)

        testset = VolumetricImageDataset(data_root_dir=config.validation_data_path,
                                         data=get_image_list(config.validation_data_file),
                                         image_size=config.image_size,
                                         network_type=config.network_type,
                                         num_images_limit=config.num_images_limit,
                                         augment=False)

        return torch.utils.data.DataLoader(trainset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           num_workers=config.num_workers,
                                           collate_fn=concatenate_samples), \
               torch.utils.data.DataLoader(testset,
                                           batch_size=config.test_batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers,
                                           collate_fn=concatenate_samples)

    else:
        raise ValueError("Unsupported dataset!")


def concatenate_samples(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    label_binary = [item[2] for item in batch]

    return [torch.cat(data, dim=0), torch.cat(label, dim=0), torch.cat(label_binary, dim=0)]


class VolumetricImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_dir, data, image_size, network_type, num_images_limit, augment, shift=10, rotate=10,
                 scale=0.1):
        self.data_root_dir = data_root_dir
        self.data = data
        self.image_size = image_size
        self.network_type = network_type
        self.num_images_limit = num_images_limit
        self.augment = augment
        self.shift = shift
        self.rotate = rotate
        self.scale = scale

    def __getitem__(self, index):
        image_name = self.data['image_filenames'][index]
        label_name = self.data['label_filenames'][index]
        phase_number = self.data['phase_numbers'][index]

        image_id = os.path.basename(image_name).split('.')[0]
        info = {}
        info['Name'] = image_id
        info['PhaseNumber'] = phase_number

        nib_image = nib.load(os.path.join(self.data_root_dir, image_name))
        info['PixelResolution'] = nib_image.header['pixdim'][1:4]
        info['ImageSize'] = nib_image.header['dim'][1:4]
        info['AffineTransform'] = nib_image.header.get_best_affine()
        info['Header'] = nib_image.header.copy()

        whole_image = nib_image.get_data()

        label_name = image_name[:-14] + 'GT_contours.json'

        with open(os.path.join(self.data_root_dir, label_name), 'r') as f:
            all_contours = json_tricks.load(f)

        num_slices = whole_image.shape[2]
        num_points_per_contour = 100
        structure_list = ['lv_endo', 'lv_epi', 'rv_endo']

        whole_contour = np.zeros([num_points_per_contour, 2, num_slices, len(structure_list)])
        whole_contour_binary = np.zeros([num_slices, len(structure_list)])

        for slice_num in range(whole_image.shape[2]):
            for j, structure in enumerate(structure_list):
                whole_contour[:, :, slice_num, j], whole_contour_binary[slice_num, j] = self.get_contour(all_contours,
                                                                                                         slice_num,
                                                                                                         structure,
                                                                                                         num_points_per_contour)

        if np.ndim(whole_image) == 4:
            whole_image = whole_image[:, :, :, phase_number]

        clip_min = np.percentile(whole_image, 1)
        clip_max = np.percentile(whole_image, 99)
        whole_image = np.clip(whole_image, clip_min, clip_max)
        whole_image = (whole_image - whole_image.min()) / float(whole_image.max() - whole_image.min())

        whole_contour = np.roll(whole_contour, shift=1, axis=1)

        x, y, z = whole_image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        cropped_image, crop_offset_x, crop_offset_y = image_utils.crop_image(whole_image, x_centre, y_centre, [self.image_size, self.image_size])
        cropped_contour = whole_contour - np.expand_dims(np.expand_dims(np.array([[crop_offset_x, crop_offset_y]]), axis=-1), axis=-1)

        # Perform data augmentation
        if self.augment and (np.random.rand() > 0.33):
            if self.network_type == '2d' or self.network_type == 'bayesian2d' or self.network_type == 'bayesian2d_mc_dropout':
                cropped_image, cropped_contour = image_utils.augment_image_contour_2d(cropped_image, cropped_contour,
                                                                           preserve_across_slices=False,
                                                                           max_shift=self.shift, max_rotate=self.rotate,
                                                                           max_scale=self.scale)
            elif self.network_type == '2.5d':
                cropped_image, cropped_contour = image_utils.augment_image_contour_2d(cropped_image, cropped_contour,
                                                                           preserve_across_slices=True,
                                                                           max_shift=self.shift, max_rotate=self.rotate,
                                                                           max_scale=self.scale)
            else:
                raise Exception("Unknown type in data augmentation.")

        if self.network_type == '2d' or self.network_type == 'bayesian2d' or self.network_type == 'bayesian2d_mc_dropout':
            # Put into NHWC format
            batch_images = np.expand_dims(np.transpose(cropped_image, axes=(2, 0, 1)), axis=-1)
            batch_contours = np.transpose(cropped_contour, axes=(2, 0, 1, 3))
            batch_contours_binary = whole_contour_binary

            if batch_images.shape[0] > self.num_images_limit:
                slices = sorted(list(np.random.permutation(batch_images.shape[0]))[:self.num_images_limit])
                batch_images = batch_images[slices, :, :, :]
                batch_contours = batch_contours[slices, :, :]
                batch_contours_binary = batch_contours_binary[slices, :]
                # print("Warning: Number of slices limited to {} to fit GPU".format(num_images_limit))

        else:
            raise Exception("Unknown type in get_batch.")

        return torch.Tensor(batch_images), torch.Tensor(batch_contours), torch.Tensor(batch_contours_binary)

    def get_contour(self, all_contours, slice_num, structure, num_points_per_contour):
        if ('Slice{}'.format(slice_num) in all_contours) and (structure in all_contours['Slice{}'.format(slice_num)]):
            contour = all_contours['Slice{}'.format(slice_num)][structure]

            # Remove extra segments
            first_point = contour[0, :]

            repeat_indices = np.where(np.linalg.norm(contour - first_point, axis=1) == 0)

            if len(repeat_indices[0]) > 1:
                if repeat_indices[0][1] != (len(contour) - 1):
                    # print(repeat_indices[0][1], len(contour))
                    contour = contour[:repeat_indices[0][1] + 1, :]
                    # print('--------Removed points at the end of contour')
            else:
                contour = np.concatenate([contour, np.expand_dims(first_point, axis=0)], axis=0)

            # Remove consecutive repeated points
            num_points = contour.shape[0]
            include_points = np.ones([num_points], dtype=np.bool)

            for i in range(num_points - 1):
                if np.linalg.norm(contour[i] - contour[i + 1]) == 0:
                    include_points[i + 1] = False

            contour = contour[include_points]

            spline_order = min(3, len(contour[:, 0]) - 1)

            try:
                tck, u = interpolate.splprep([contour[:, 0], contour[:, 1]], s=1, k=spline_order)
            except:
                print('Countour shape: ', contour.shape)
                print('Spline order: ', spline_order)
                print('Slice num: ', slice_num)
                print('Structure: ', structure)
                plt.plot(contour[:,1], contour[:,0], '.')
                plt.savefig('error_contour.png')
                np.save('error_contour.npy', contour)
                raise Exception('Error in scipy interpolation')

            contour_resampled = np.zeros([num_points_per_contour, 2])
            [contour_resampled[:, 0], contour_resampled[:, 1]] = interpolate.splev(np.linspace(0, 1, num_points_per_contour), tck)

            contour_binary = 1

        else:
            contour_resampled = np.zeros([num_points_per_contour, 2])
            contour_binary = 0

        return contour_resampled, contour_binary

    def __len__(self):
        return len(self.data['image_filenames'])


def get_image_list(csv_file):
    image_list, label_list, phase_list = [], [], []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            phase_numbers = eval(row['phase_numbers'])

            for phase in phase_numbers:
                image_list.append(row['image_filenames'].strip())
                label_list.append(row['label_filenames'].strip())
                phase_list.append(phase)

    data_list = {}
    data_list['image_filenames'] = image_list
    data_list['label_filenames'] = label_list
    data_list['phase_numbers'] = phase_list

    return data_list
