import torch
import torch.nn as nn
from .resnet_GCN_skip import SkipResnet50
from .first_annotation import FirstAnnotation
from .detection_network import DetectionNetwork
from .edge_graph_nn import EdgeGraphNN
from .multiscale_edge_graph_nn import MultiscaleEdgeGraphNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolyGNN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 n_adj=6,
                 coarse_to_fine_steps=0,
                 get_point_annotation=False):

        super(PolyGNN, self).__init__()

        # Modified from https://github.com/fidler-lab/curve-gcn/code/Models/GNN/poly_gnn.py
        pass


    def forward(self, x, init_polys1, init_polys2, init_polys3, num_polys):
        # Modified from https://github.com/fidler-lab/curve-gcn/code/Models/GNN/poly_gnn.py
        pass


    def reload(self, path, strict=False):
        print("Reloading full model from: ", path)
        self.load_state_dict(torch.load(path)['state_dict'], strict=strict)
