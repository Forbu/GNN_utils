""" Description: setup.py for GNN_utils package
"""
import setuptools

# create a setuptools setup that will name the package torch_geometric
# and will include all the files in the torch_geometric directory
# and that will have a dependency on torch and torch_scatter
setuptools.setup(
    name="gnn_utils",
    version="1.6.3",
    packages=setuptools.find_packages(),
    install_requires=["torch", "torch_scatter"],
)
