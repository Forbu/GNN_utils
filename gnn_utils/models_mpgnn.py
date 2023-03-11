"""
In this script we will create a simple GNN model for the task of node regression.
With the help of the PyTorch Geometric library.
"""
# main torch imports
import torch
from torch import Tensor

import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing

from gnn_utils.utils import MLP, get_blocks_encoding_decoding


class MPGNNConv(MessagePassing):
    """
    Simple layer for message passing

    """

    def __init__(self, node_dim, edge_dim, layers=3):
        super().__init__(aggr="mean", node_dim=0)
        self.lin_edge = MLP(
            in_dim=node_dim * 2 + edge_dim, out_dim=edge_dim, hidden_layers=layers
        )
        self.lin_node = MLP(
            in_dim=node_dim + edge_dim, out_dim=node_dim, hidden_layers=layers
        )

    def forward(self, node, edge_index, edge_attr):
        """
        here we apply the message passing function
        and then we apply the MLPs to the output of the message passing function
        """
        init_node = node

        # message passing
        message_info = self.propagate(edge_index, x=node, edge_attr=edge_attr)

        # we concat the output of the message passing function with the input node features
        node = torch.cat((node, message_info), dim=-1)

        # now we apply the MLPs with residual connections
        node = self.lin_node(node) + init_node

        return node, edge_attr

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor):
        """
        Message function for the message passing
        Basically we concatenate the node features and the edge features

        Args:
            x_j (Tensor): Tensor of shape (E, node_dim) where E is the number of edges. FloatTensor
            x_i (Tensor): Tensor of shape (E, node_dim) where E is the number of edges. FloatTensor
            edge_attr (Tensor): Tensor of shape (E, edge_dim) where E is the number of edges. FloatTensor

        """
        edge_info = torch.cat((x_i, x_j, edge_attr), dim=-1)

        print("edge_info", edge_info.shape)

        edge_info = self.lin_edge(edge_info)
        return edge_info


class MPGNN(pl.LightningModule):
    """Graph Network-based Simulators(GNS)"""

    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10,  # number of GNN layers
        node_feature_dim=30,
        edge_feature_dim=3,
        dim=2,  # output dimension
    ):
        super().__init__()

        graph_encoders, self.node_in, self.node_out = get_blocks_encoding_decoding(
            nb_graphs=1,
            input_dim_node=node_feature_dim,
            input_dim_edges=[edge_feature_dim],
            hidden_dim=hidden_size,
            output_dim=dim,
            hidden_dim_edge=hidden_size,
        )

        self.edge_in = graph_encoders[0]

        self.layers = torch.nn.ModuleList(
            [
                MPGNNConv(node_dim=hidden_size, edge_dim=hidden_size, layers=3)
                for _ in range(n_mp_layers)
            ]
        )

    def forward(self, edge_index, node_feature, edge_feature):
        """Forward pass of the model

        Args:
            edge_index (Tensor): Tensor of shape (2, E) where E is the number of edges. LongTensor
            node_feature (Tensor): Tensor of shape (N, node_feature_dim)
                                    where N is the number of nodes. FloatTensor
            edge_feature (Tensor): Tensor of shape (E, edge_feature_dim)
                                    where E is the number of edges. FloatTensor

        Returns:
            Tensor: Tensor of shape (N, dim) where N is the number of nodes. FloatTensor
        """
        # encoder
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(edge_feature)
        # processor
        for layer in self.layers:
            node_feature, edge_feature = layer(
                node_feature, edge_index, edge_attr=edge_feature
            )
        # decoder
        out = self.node_out(node_feature)
        return out
