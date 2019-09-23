import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FirstAnnotation(nn.Module):
    def __init__(self, feats_grid_size, in_channels, hidden_channels, out_channels):
        super(FirstAnnotation, self).__init__()
        # Modified from https://github.com/fidler-lab/curve-gcn/from code/Models/GNN/first_annotation.py
        pass


    def forward(self, feats, temperature=0.0, beam_size=1):
        # Modified from https://github.com/fidler-lab/curve-gcn/code/Models/GNN/first_annotation.py
        pass
