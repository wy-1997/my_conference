# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:02:17 2024

@author: oemn
"""


import argparse

# create a parser object
parser = argparse.ArgumentParser(description='GNN parameters')

# add parameters
parser.add_argument('--hidden_features', type=int, default=16, help='Number of hidden features')
parser.add_argument('--rho', type=float, default=0.5, help='Parameter rho')
parser.add_argument('--alpha', type=float, default=1.0, help='Parameter alpha')
parser.add_argument('--beta', type=float, default=1.0, help='Parameter beta')
parser.add_argument('--scale', type=float, default=1.0, help='Parameter scale')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--epsilon', type=float, default=1e-7, help='epsilon of normalization')
parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to use (Cora, PubMed, CiteSeer)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for NeighborSampler')
parser.add_argument('--epochs', type=int, default=100, help='Epoch for training')
parser.add_argument('--sizes', type=int, nargs='+', default=[25, 25], help='Sizes for NeighborSampler')
parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help='Device ids for multi-GPU training')
parser.add_argument('--log_dir', type=str, default='./log', help='Directory for logs')

# parse the arguments
args = parser.parse_args()