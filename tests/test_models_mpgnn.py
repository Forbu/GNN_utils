import pytest
import torch

from torch_geometric.data import Data

from pytorch_lightning.utilities.model_summary import ModelSummary

from torch_geo.models_mpgnn import MPGNNConv, MPGNN


def test_mpgnnconv():
    model = MPGNNConv(node_dim=128, edge_dim=128, layers=3)

    nb_nodes = 40000
    nb_edge = nb_nodes * 4

    # now we have to create the grid graph with 40000 nodes and 160000 edges (4 edges per node)
    edge_index_grid = torch.randint(0, nb_nodes, (2, nb_edge), dtype=torch.long)

    # we also need to create the edge attributes for the grid graph with 2 attributes per edge
    edge_attr_grid = torch.rand((nb_edge, 128), dtype=torch.float)

    # we also need to create the input features
    nodes = torch.rand((nb_nodes, 128), dtype=torch.float)

    # we convert the model to eval mode
    model.eval()

    # we check the speed
    # convert to cuda
    nodes = nodes.cuda()

    edge_attr_grid = edge_attr_grid.cuda()
    edge_index_grid = edge_index_grid.cuda()

    model = model.cuda()

    import time

    current_time = time.time()

    with torch.inference_mode():
        for _ in range(100):
            # now we can perform the forward pass
            node_output, edge_output = model(
                edge_index=edge_index_grid, node=nodes, edge_attr=edge_attr_grid
            )

    time = time.time() - current_time

    print("time for 1 forward pass: ", time / 100)

    assert node_output.shape == (nb_nodes, 128)
    assert edge_output.shape == (nb_edge, 128)

    assert False


def test_mpgnn_full():
    model = MPGNN(hidden_size=128,
        n_mp_layers=10,  # number of GNN layers
        node_feature_dim=128,
        edge_feature_dim=128,
        dim=2)  # dimension of the world, typically 2D or 3D)

    nb_nodes = 10000
    nb_edge = nb_nodes * 4

    # now we have to create the grid graph with 40000 nodes and 160000 edges (4 edges per node)
    edge_index_grid = torch.randint(0, nb_nodes, (2, nb_edge), dtype=torch.long)

    # we also need to create the edge attributes for the grid graph with 2 attributes per edge
    edge_attr_grid = torch.rand((nb_edge, 128), dtype=torch.float)

    # we also need to create the input features
    nodes = torch.rand((nb_nodes, 128), dtype=torch.float)

    # we convert the model to eval mode
    model.eval()

    # we check the speed
    # convert to cuda
    nodes = nodes.cuda()

    edge_attr_grid = edge_attr_grid.cuda()
    edge_index_grid = edge_index_grid.cuda()

    model = model.cuda()

    import time

    current_time = time.time()

    with torch.inference_mode():
        for _ in range(100):
            # now we can perform the forward pass
            output = model(
                edge_index=edge_index_grid, node_feature=nodes, edge_feature=edge_attr_grid
            )

    time = time.time() - current_time

    print("time for 1 forward pass: ", time / 100)

    print(ModelSummary(model, max_depth=3))

    assert False
