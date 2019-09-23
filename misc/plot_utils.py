import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import metrics


def plot_merged_images(x, pred, gt, pred_binary, gt_binary, save_dir=None, filename=None):
    '''

    :param x: (N, 1, H, W, 1) numpy array
    :param pred: (N, num_contour_points, 2, num_structures) numpy array
    :param gt: (N, num_contour_points, 2, num_structures) numpy array
    :param pred_binary: (N, num_structures) numpy array
    :param gt_binary: (N, num_structures) numpy array
    :return:
    '''
    n = x.shape[0]

    plt.figure(figsize=(10, 10))

    colors = ['red', 'blue', 'green']

    for i in range(n):
        ax = plt.subplot(3, 4, i+1)

        ax.imshow(x[i, 0, :, :], cmap='gray', interpolation='none')

        for s in range(pred.shape[-1]):
            if pred_binary[i, s] == 1:
                ax.plot(pred[i, :, 1, s], pred[i, :, 0, s], color=colors[s], linestyle='-', linewidth=0.5)

            if gt_binary[i, s] == 1:
                ax.plot(gt[i, :, 1, s], gt[i, :, 0, s], 'r-', color=colors[s], linestyle=':', linewidth=0.5)

            # Plot first point and second point
            # ax.plot(pred[i, 0, 1], pred[i, 0, 0], 'g.', markersize=0.05)
            # ax.plot(pred[i, 1, 1], pred[i, 1, 0], 'y.', markersize=0.05)

            # ax.plot(gt[i, 0, 1], gt[i, 0, 0], 'g.', markersize=0.05)
            # ax.plot(gt[i, 1, 1], gt[i, 1, 0], 'y.', markersize=0.05)

    if save_dir is not None and filename is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, filename))

    plt.close()

    return


def plot_images_and_contour_evolution(x, pred, gt, pred_binary, gt_binary, save_dir=None, filename=None):
    '''

    :param x: (N, 1, H, W, 1) numpy array
    :param pred: (N, num_contour_points, 2, num_iterations, num_structures) numpy array
    :param gt: (N, num_contour_points, 2, num_structures) numpy array
    :param pred_binary: (N, num_structures) numpy array
    :param gt_binary: (N, num_structures) numpy array
    :return:
    '''
    n = x.shape[0]

    plt.figure(figsize=(10, 10))

    colors = [['#fcae91','#fb6a4a','#de2d26','#a50f15'],   # Structure 1 - Shades of red
              ['#bdd7e7','#6baed6','#3182bd','#08519c'],   # Structure 2 - Shades of blue
              ['#bae4b3','#74c476','#31a354','#006d2c']]   # Structure 3 - Shades of green

    for i in range(n):
        ax = plt.subplot(3, 4, i+1)

        ax.imshow(x[i, 0, :, :], cmap='gray', interpolation='none')

        for s in range(pred.shape[-1]):
            if pred_binary[i, s] == 1:

                for t in [-2, -1]:
                    ax.plot(pred[i, :, 1, t, s], pred[i, :, 0, t, s], color=colors[s][t], linestyle='-', linewidth=0.5)

            if gt_binary[i, s] == 1:
                ax.plot(gt[i, :, 1, s], gt[i, :, 0, s], 'r-', color=colors[s][-1], linestyle=':', linewidth=0.5)

            # Plot first point and second point
            # ax.plot(pred[i, 0, 1], pred[i, 0, 0], 'g.', markersize=0.05)
            # ax.plot(pred[i, 1, 1], pred[i, 1, 0], 'y.', markersize=0.05)

            # ax.plot(gt[i, 0, 1], gt[i, 0, 0], 'g.', markersize=0.05)
            # ax.plot(gt[i, 1, 1], gt[i, 1, 0], 'y.', markersize=0.05)

    if save_dir is not None and filename is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, filename))

    plt.close()

    return


def plot_contours_and_mask(x, pred, gt, save_dir=None, filename=None):
    '''
    :param x: (N, 1, H, W) numpy array
    :param pred: (N, num_contour_points, 2) numpy array
    :param gt: (N, num_contour_points, 2) numpy array
    :return:
    '''
    n, _, h, w = x.shape

    pred_mask = metrics.mask_from_poly(pred, w, h, upsampling_factor=10)
    gt_mask = metrics.mask_from_poly(gt, w, h, upsampling_factor=10)

    plt.figure(figsize=(10, 10))

    for i in range(n):
        ax = plt.subplot(3, 4, i+1)

        ax.imshow(x[i, 0, :, :], cmap='gray', interpolation='none')
        # ax.plot(pred[i, :, 1], pred[i, :, 0], 'b-', linewidth=0.5)
        ax.plot(gt[i, :, 1], gt[i, :, 0], 'r-', linewidth=0.5)

        # ax.imshow(pred_mask[i, :, :].transpose(), cmap='jet', interpolation='none', alpha=0.5)
        ax.imshow(gt_mask[i, :, :].transpose(), cmap='jet', interpolation='none', alpha=0.5)

    if save_dir is not None and filename is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, filename))

    plt.close()

    return
