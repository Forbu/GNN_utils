"""
Simple testing session for the MPGNN model
"""
import time
import pytest
import torch

from pytorch_lightning.utilities.model_summary import ModelSummary
from torch_geo.models_mpgnn import MPGNNConv, MPGNN

# fixture to initialize the test with a layer
@pytest.fixture(scope="module", autouse=True, name="layer")
def create_layer():
    """
    Layer to test
    """
    layer = MPGNNConv(node_dim=128, edge_dim=128, layers=3)
    layer.eval()
    return layer

# fixture to initialize the test with a model
@pytest.fixture(scope="module", autouse=True, name="model")
def create_model():
    """
    Model to test
    """
    model_gnn = MPGNN(
        node_feature_dim=30,
        edge_feature_dim=3,
        hidden_size=128,
        dim=128,
        n_mp_layers=10,
    )
    model_gnn.eval()
    return model_gnn


# fixture to initialize the test input
@pytest.fixture(scope="module", autouse=True, name="inputs_model")
def initialize_input():
    """
    We create a grid graph with 40000 nodes and 160000 edges (4 edges per node)
    It is a simple grid graph with 200x200 nodes
    """
    nb_nodes = 40000
    nb_edge = nb_nodes * 4

    # now we have to create the grid graph with 40000 nodes and 160000 edges (4 edges per node)
    edge_index_grid = torch.randint(0, nb_nodes, (2, nb_edge), dtype=torch.long)

    # we also need to create the edge attributes for the grid graph with 2 attributes per edge
    edge_attr_grid = torch.rand((nb_edge, 128), dtype=torch.float)

    # we also need to create the input features
    nodes = torch.rand((nb_nodes, 128), dtype=torch.float)

    return edge_index_grid, edge_attr_grid, nodes, nb_nodes, nb_edge


def test_mpgnnconv(layer, inputs_model):
    """
    Testing of the MPGNNConv model (simple forward pass)
    """

    edge_index_grid, edge_attr_grid, nodes, nb_nodes, nb_edge = inputs_model

    # we check the speed
    # convert to cuda
    nodes = nodes.cuda()

    edge_attr_grid = edge_attr_grid.cuda()
    edge_index_grid = edge_index_grid.cuda()

    model = layer.cuda()

    current_time = time.time()

    with torch.inference_mode():
        for _ in range(100):
            # now we can perform the forward pass
            node_output, edge_output = model(
                edge_index=edge_index_grid, node=nodes, edge_attr=edge_attr_grid
            )

    time_duration = time.time() - current_time

    print("time for 1 forward pass: ", time_duration / 100)

    assert node_output.shape == (nb_nodes, 128)
    assert edge_output.shape == (nb_edge, 128)


def test_mpgnn_full(model, inputs_model):
    """
    Testing of the MPGNN model and also we try to print the summary of the model
    and the time for 1 forward pass
    """

    edge_index_grid, edge_attr_grid, nodes, nb_nodes, _ = inputs_model

    # we check the speed
    # convert to cuda
    nodes = nodes.cuda()

    edge_attr_grid = edge_attr_grid.cuda()
    edge_index_grid = edge_index_grid.cuda()

    model = model.cuda()

    current_time = time.time()

    with torch.inference_mode():
        for _ in range(100):
            # now we can perform the forward pass
            output = model(
                edge_index=edge_index_grid,
                node_feature=nodes,
                edge_feature=edge_attr_grid,
            )

    time_duration = time.time() - current_time

    print("time for 1 forward pass: ", time_duration / 100)

    print(ModelSummary(model, max_depth=3))

    assert output.shape == (nb_nodes, 128)
