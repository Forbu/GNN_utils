"""
Utility functions for GNNs
"""

from torch import nn
from torch_geometric.nn import GATv2Conv


class MLP(nn.Module):
    """
    Simple MLP (multi-layer perceptron)
    """

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
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', ...
        """

        super().__init__()

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

    def forward(self, vector):
        """
        Simple forward pass
        """
        return self.model(vector)


def get_blocks_encoding_decoding(
    nb_graphs,
    input_dim_node,
    input_dim_edges,
    hidden_dim,
    output_dim,
    hidden_dim_edge=None,
):
    """
    Returns the encoding and decoding blocks for a GNN
    """
    assert (
        len(input_dim_edges) == nb_graphs
    ), "input_dim_edges should be a list of length nb_graphs"

    # encoder
    graph_encoders = nn.ModuleList()
    for i in range(nb_graphs):
        graph_encoders.append(
            MLP(
                in_dim=input_dim_edges[i],
                out_dim=hidden_dim_edge,
                hidden_dim=hidden_dim,
                hidden_layers=2,
            )
        )

    # encoder and decoder for the node features
    node_encoder = MLP(
        in_dim=input_dim_node,
        out_dim=hidden_dim,
        hidden_dim=hidden_dim,
        hidden_layers=2,
    )

    node_decoder = MLP(
        in_dim=hidden_dim,
        out_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=2,
    )

    return graph_encoders, node_encoder, node_decoder


def get_blocks_message_passing(nb_graph, hidden_dim, nb_head=2, nb_iterations=10):
    """
    Returns the message passing blocks for a GNN

    Args:
        nb_graph: number of graphs
        hidden_dim: hidden dimension
        nb_head: number of heads (for the GATv2Conv)
        nb_iterations: number of iterations (message passing)

    Returns:
        graphs_message_passing: list of list of GATv2Conv
        nodes_message_passing: list of MLP that will be applied
                            to the nodes after the message passing
    """

    # init the message passing blocks for each graph
    graphs_message_passing = nn.ModuleList()

    for _ in range(nb_graph):
        # init the message passing blocks for each iteration
        message_passing = nn.ModuleList()
        for _ in range(nb_iterations):
            message_passing.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // nb_head,
                    heads=nb_head,
                    concat=True,
                    edge_dim=hidden_dim // nb_head,
                )
            )
        graphs_message_passing.append(message_passing)

    nodes_message_passing = nn.ModuleList()

    # now we can create the nodes preprocessing
    for _ in range(nb_iterations):
        nodes_message_passing.append(
            MLP(in_dim=hidden_dim * nb_graph, out_dim=hidden_dim)
        )

    return graphs_message_passing, nodes_message_passing
