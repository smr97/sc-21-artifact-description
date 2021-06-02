#  Dynamic Graph Neural Networks Scaling SC 21 artifact description
Contains the artifact description for our paper on scaling GNNs
We are unable to release the source code as it is classified as proprietary by the organization at this stage.
This README contains as much detail as possible for someone to be able to implement the scaling aspects discussed in the paper.

The models are available as follows:

EvolveGCN: https://github.com/IBM/EvolveGCN

TM-GCN: https://github.com/IBM/TM-GCN

CD-GCN: Implementation has been done from the specifications in the paper "F. Manessi, A. Rozza, and M. Manzo. Dynamic graph convolutional net-works. Pattern Recognition (2020).". The specifications are very clear for implementation purposes.

The datasets are all publicly available:

flickr: http://networkrepository.com/soc-flickr-growth.php

youtube: http://networkrepository.com/soc-youtube-growth.php

epinions: http://networkrepository.com/rec-epinions-user-ratings.php

AMLSim: https://github.com/IBM/AMLSim

Random data: The file `sparse_tensor_creator.py` has been attached in this repository.

For each model, the distribution strategy has been implemented as-is from the paper.

The experimental setup has been clearly defined in the paper:
The experiments were conducted on a system having 16 nodes, each with 8 GPUs, leading to a total of 128 GPUs. Each node has 2x20 cores of 2.5GHz Intel Xeon Gold 6248 and each GPU is NVIDIA Tesla V100 with 32 GiB HBM and 768 GiB RAM. The nodes are connected by Dual 100 Gb EDR Infiniband. In each node, we run up to eight processes, each controlling a single GPU. We map these processes to separate cores of the node. We use Pytorch 1.7.1 for training, Nccl 2.8.4 for backend communication and Pynccl 0.1.2 for collective routines. All our codes are implemented in python.
