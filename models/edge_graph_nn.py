import torch
import torch.nn as nn
import utils.utils as utils

from .GCN import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EdgeGraphNN(nn.Module):
    def __init__(self, encoder_final_grid_size, encoder_final_feats_channels, gnn_state_dim, coarse_to_fine_steps, n_adj):
        super(EdgeGraphNN, self).__init__()
        # Modified from https://github.com/fidler-lab/curve-gcn/code/Models/GNN/poly_gnn.py
        pass


    def forward(self, x, init_polys):
        # Modified from https://github.com/fidler-lab/curve-gcn/code/Models/GNN/poly_gnn.py
        pass


    def sampling(self, coords, features):
        # Copied from https://github.com/fidler-lab/curve-gcn/code/Models/Encoder/resnet_GCN_skip.py
        pass


    def interpolated_sum(self, features, coords, grid_size):
        # Copied from https://github.com/fidler-lab/curve-gcn/code/Models/GNN/poly_gnn.py
        pass