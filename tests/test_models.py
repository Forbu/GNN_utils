"""
Test module for the models (the GAT model and the RayTracing model)
"""
import pytest
import time

import torch
from torch_geometric.data import Data
from pytorch_lightning.utilities.model_summary import ModelSummary

from torch_geo.models import RayTracingModel, RayTracingModelGAT

# pytest to initialize the test with a model
@pytest.fixture(scope="module", autouse=True)
def model():
    """
    Returns:
       nn.Module: RayTracingModel
    """
    model = RayTracingModel(
        nb_iterations=10,
        input_dim=5,
        output_dim=5,
        hidden_dim=128,
        graph_grid_atrr_dim=2,
        graph_ray_atrr_dim=1,
    )

    model.eval()

    return model


# another pytest to initialize the test input
@pytest.fixture(scope="module", autouse=True)
def initialize_input():
    """
    Fixture to initialize the input for the model
    """
    nb_nodes = 10000
    nb_edge = nb_nodes * 4

    # now we have to create the grid graph with 40000 nodes and 160000 edges (4 edges per node)
    edge_index_grid = torch.randint(0, nb_nodes, (2, nb_edge), dtype=torch.long)

    # we also need to create the edge attributes for the grid graph with 2 attributes per edge
    edge_attr_grid = torch.rand((nb_edge, 128), dtype=torch.float)

    # we do the same for the ray graph
    edge_index_ray = torch.randint(0, nb_nodes, (2, nb_edge), dtype=torch.long)

    # we also need to create the edge attributes for the grid graph with 2 attributes per edge
    edge_attr_ray = torch.rand((nb_edge, 1), dtype=torch.float)

    # we also need to create the input features
    nodes = torch.rand((nb_nodes, 128), dtype=torch.float)

    return (
        edge_index_grid,
        edge_attr_grid,
        edge_index_ray,
        edge_attr_ray,
        nodes,
        nb_nodes,
    )


@pytest.fixture(scope="module", autouse=True)
def test_simple_model(model, initialize_input):
    """
    Simple test to check if the model is working
    """
    (
        edge_index_grid,
        edge_attr_grid,
        edge_index_ray,
        edge_attr_ray,
        nodes,
        nb_nodes,
    ) = initialize_input

    with torch.inference_mode():

        output = model(
            edge_index_grid, edge_attr_grid, edge_index_ray, edge_attr_ray, nodes
        )

    assert output.shape == (nb_nodes, 5)


def convert_to_cuda(
    edge_index_grid,
    edge_attr_grid,
    edge_index_ray,
    edge_attr_ray,
    nodes,
    nb_nodes,
):
    """
    Function to convert the input to cuda
    """
    # convert to cuda
    nodes = nodes.cuda()

    edge_attr_grid = edge_attr_grid.cuda()
    edge_attr_ray = edge_attr_ray.cuda()

    edge_index_grid = edge_index_grid.cuda()
    edge_index_ray = edge_index_ray.cuda()

    return (
        edge_index_grid,
        edge_attr_grid,
        edge_index_ray,
        edge_attr_ray,
        nodes,
        nb_nodes,
    )


def test_speed_model(model, initialize_input):
    """
    test to check the speed of the model on a full graph
    """

    (
        edge_index_grid,
        edge_attr_grid,
        edge_index_ray,
        edge_attr_ray,
        nodes,
        nb_nodes,
    ) = initialize_input

    # convert to cuda
    (
        edge_index_grid,
        edge_attr_grid,
        edge_index_ray,
        edge_attr_ray,
        nodes,
        nb_nodes,
    ) = convert_to_cuda(
        edge_index_grid, edge_attr_grid, edge_index_ray, edge_attr_ray, nodes, nb_nodes
    )

    current_time = time.time()

    with torch.inference_mode():
        for _ in range(100):
            # now we can perform the forward pass
            output = model(
                edge_index_grid, edge_attr_grid, edge_index_ray, edge_attr_ray, nodes
            )

    time = time.time() - current_time

    print("time for 1 forward pass: ", time / 100)
    print(ModelSummary(model, max_depth=3))

