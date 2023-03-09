"""
In this script we will create a simple GNN model for the task of node regression.
With the help of the PyTorch Geometric library.

This script will implement a GNN model that will take 4 graphs as input:
- the first one is the "grid" graph
- the second one is the "ray" graph
- the third / fourth one are the "grid" graph at the next scale (the next scale is a 4x4 grid)
"""
# main torch imports
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn

import pytorch_lightning as pl

# torch geometric imports
import torch_geometric.transforms as T

from torch_geometric.nn.conv.nn_conv import NNConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv


class MLP(nn.Module):
    # MLP with LayerNorm
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):
        """
        MLP
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RayTracingMultiScaleModelGAT(pl.LightningModule):
    """
    GNN model :
    The idea of this gnn model is to take 2 graphs structure as input :
    - the first one is the "grid" graph
    - the second one is the "ray" graph

    We will use the GATv2 layer to combine the 2 graphs.

    We also use the multi-scale approach to combine the 2 graphs at different scales.
    https://arxiv.org/pdf/2210.00612.pdf

    """

    def __init__(
        self,
        nb_iterations=10,
        input_dim=5,
        output_dim=5,
        hidden_dim=128,
        graph_grid_atrr_dim=2,
        graph_ray_atrr_dim=1,
        nb_head=2,
    ) -> None:
        super().__init__()

        self.nb_iterations = nb_iterations

        # here we first create the encoder MLP for the grid graph and the ray graph
        self.encoder_grid = MLP(in_dim=graph_grid_atrr_dim, out_dim=hidden_dim // 2)
        self.encoder_ray = MLP(in_dim=graph_ray_atrr_dim, out_dim=hidden_dim // 2)

        # encoder for the input features
        self.node_encoder = MLP(in_dim=input_dim, out_dim=hidden_dim)

        # then we have a final decoder MLP
        self.decoder = MLP(in_dim=hidden_dim, out_dim=output_dim)

        # then we create nb_iterations NNConv layers for the message passing for the grid graph and the ray graph
        self.message_passing_grid = nn.ModuleList()
        self.message_passing_ray = nn.ModuleList()
        self.message_passing_grid_multiscale = nn.ModuleList()

        self.node_preprocessing = nn.ModuleList()

        for _ in range(nb_iterations):
            self.message_passing_grid.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // 2,
                    heads=nb_head,
                    concat=True,
                    edge_dim=hidden_dim // 2,
                )
            )
            self.message_passing_ray.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // 2,
                    heads=nb_head,
                    concat=True,
                    edge_dim=hidden_dim // 2,
                )
            )
            self.message_passing_grid_multiscale.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // 2,
                    heads=nb_head,
                    concat=True,
                    edge_dim=hidden_dim // 2,
                )
            )

            self.node_preprocessing.append(
                MLP(in_dim=hidden_dim * nb_head, out_dim=hidden_dim)
            )

        # initialize the multi-scale graph node
        self.multiscale_node = torch.nn.Parameter(torch.randn(1, hidden_dim))

    def forward(
        self,
        grid_graph_edge_index,
        grid_graph_edge_attr,
        ray_graph_edge_index,
        ray_graph_edge_attr,
        x,
        multiscale_grid_graph_edge_index,
        multiscale_grid_graph_edge_attr,
        nb_multiscale_node,
    ):
        """
        Forward pass of the model

        This is a multigraph model / and also a multi-scale model.

        The multi-scale model is used to encode the multiscale grid graph.

        Args:
            grid_graph_edge_index (torch.Tensor): (2, E) tensor, where E is the number of edges in the grid graph
            grid_graph_edge_attr (torch.Tensor): (E, F) tensor, where F is the number of features for each edge in the grid graph
            ray_graph_edge_index (torch.Tensor): (2, E) tensor, where E is the number of edges in the ray graph
            ray_graph_edge_attr (torch.Tensor): (E, F) tensor, where F is the number of features for each edge in the ray graph
            x (torch.Tensor): (N, F') tensor, where N is the number of nodes in the input graph, and F is the number of input features per node.
            multiscale_grid_graph_edge_index (torch.Tensor): (2, E) tensor, where E is the number of edges in the multiscale grid graph
            multiscale_grid_graph_edge_attr (torch.Tensor): (E, F) tensor, where F is the number of features for each edge in the multiscale grid graph
            nb_multiscale_node (int): number of nodes in the multiscale grid graph

        Returns:
            torch.Tensor: (N, C) tensor, where C is the number of output.
        """
        # first we encode the grid graph and the ray graph
        x_grid_attr = self.encoder_grid(grid_graph_edge_attr)
        x_ray_attr = self.encoder_ray(ray_graph_edge_attr)

        # then we encode the input features
        x = self.node_encoder(x)

        # we initialize the multiscale node
        node_multiscale = self.multiscale_node.repeat(nb_multiscale_node, 1)

        # then we perform the message passing
        for i in range(self.nb_iterations):
            x_grid = self.message_passing_grid[i](x, grid_graph_edge_index, x_ray_attr)
            x_ray = self.message_passing_ray[i](x, ray_graph_edge_index, x_grid_attr)

            init = x

            # we also apply a node preprocessing MLP
            x = self.node_preprocessing[i](torch.cat([x_grid, x_ray], dim=1))

            # preprocess the multiscale node to get information from the grid graph
            # step 1. concatenate the multiscale node with the grid node
            x_multiscale = torch.cat([x, node_multiscale], dim=0)
            x_multiscale = self.message_passing_grid_multiscale[i](
                x_multiscale,
                multiscale_grid_graph_edge_index,
                multiscale_grid_graph_edge_attr,
            )

            node_multiscale = x_multiscale[-nb_multiscale_node:]
            x_multiscale_real = x_multiscale[:-nb_multiscale_node]

            x = x + init + x_multiscale_real

        # finally we decode the output
        x = self.decoder(x)

        return x
