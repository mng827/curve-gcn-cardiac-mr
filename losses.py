import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def polygon_matching_loss(pnum, pred, gt, loss_type, gt_binary=None):
    # Modified from https://github.com/fidler-lab/curve-gcn/code/Evaluation/losses.py
    pass


def mse_loss(input, target, gt_binary=None):
    '''
    :param input: (N, num_points_per_contour, 2)
    :param target: (N, num_points_per_contour, 2)
    :return:
    '''
    batch_size = input.size()[0]

    if gt_binary is None:
        gt_binary = torch.ones(batch_size, 1)

    return torch.mean((input[gt_binary == 1] - target[gt_binary == 1]) ** 2)


def assd_loss(input, target):
    '''
    :param input: (N, num_points_per_contour, 2)
    :param target: (N, num_points_per_contour, 2)
    :return:
    '''
    # raise Exception('ASSD Loss needs to be tested')

    dist_matrix = torch.sum(input ** 2, dim=2, keepdim=True) \
                  + torch.sum(target ** 2, dim=2, keepdim=True).transpose(2,1) \
                  - 2 * torch.bmm(input, target.transpose(2,1))

    dist_matrix = torch.sqrt(dist_matrix)
    print(dist_matrix)

    assd = 0.5 * (torch.mean(torch.min(dist_matrix, dim=1)[0]) + torch.mean(torch.min(dist_matrix, dim=2)[0]))
    return assd


def binary_classification_loss(pred_logits, target):
    return F.binary_cross_entropy_with_logits(pred_logits, target)


if __name__ == '__main__':
    a = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    b = torch.tensor([[[5, 6], [7, 8], [9, 10]]], dtype=torch.float32)

    print(a.shape)
    print(b.shape)
    print(assd_loss(a, b))
