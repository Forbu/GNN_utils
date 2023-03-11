"""
In this script we will create a simple GNN model for the task of node regression.
With the help of the PyTorch Geometric library.
"""
# main torch imports
import torch

import pytorch_lightning as pl

from gnn_utils.utils import (
    get_blocks_encoding_decoding,
    get_blocks_message_passing,
)

class RayTracingModelGAT(pl.LightningModule):
    """
    GNN model :
    The idea of this gnn model is to take 2 graphs structure as input :
    - the first one is the "grid" graph
    - the second one is the "ray" graph

    We will use the GAT layer to combine the 2 graphs.
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

        # get the blocks for the encoding and the decoding
        graph_encoders, self.node_encoder, self.decoder = get_blocks_encoding_decoding(
            nb_graphs=2,
            input_dim_node=input_dim,
            input_dim_edges=[graph_grid_atrr_dim, graph_ray_atrr_dim],
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim_edge=hidden_dim // nb_head,
        )

        # here we first create the encoder MLP for the grid graph and the ray graph
        self.encoder_grid = graph_encoders[0]
        self.encoder_ray = graph_encoders[1]

        graphs_message_passing, nodes_message_passing = get_blocks_message_passing(
            nb_graph=2,
            hidden_dim=hidden_dim,
            nb_head=nb_head,
            nb_iterations=nb_iterations,
        )

        # the message passing for the grid graph and the ray graph
        self.message_passing_grid = graphs_message_passing[0]
        self.message_passing_ray = graphs_message_passing[1]

        self.node_preprocessing = nodes_message_passing

    def forward(
        self,
        grid_graph_edge_index,
        grid_graph_edge_attr,
        ray_graph_edge_index,
        ray_graph_edge_attr,
        node_features,
    ):
        """
        Forward pass of the model
        It follows the same structure as the previous model :
        - first we encode the grid graph and the ray graph
        - then we encode the input features
        - then we perform the message passing
        - finally we decode the output

        """
        # first we encode the grid graph and the ray graph
        x_grid_attr = self.encoder_grid(grid_graph_edge_attr)
        x_ray_attr = self.encoder_ray(ray_graph_edge_attr)

        # then we encode the input features
        node_features = self.node_encoder(node_features)

        # then we perform the message passing
        for i in range(self.nb_iterations):
            
            x_grid = self.message_passing_grid[i](
                node_features, grid_graph_edge_index, x_ray_attr
            )
            x_ray = self.message_passing_ray[i](
                node_features, ray_graph_edge_index, x_grid_attr
            )

            init = node_features

            # we also apply a node preprocessing MLP
            node_features = self.node_preprocessing[i](
                torch.cat([x_grid, x_ray], dim=1)
            )

            node_features = node_features + init

        # finally we decode the output
        node_features = self.decoder(node_features)

        return node_features
