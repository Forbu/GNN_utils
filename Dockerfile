FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install torch geometric
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
RUN pip install torch-geometric

# Install other dependencies
RUN pip install tqdm
RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install scikit-learn

# install pandas and numpy
RUN pip install pandas
RUN pip install numpy