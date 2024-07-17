# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:08:51 2024

@author: oemn
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from GNN_layers import CustomGNNLayer2, CustomGNNLayer3, CustomGNNLayer4
from arg_parser import args
import gc
import numpy as np

class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, num_heads, alpha, beta, rho, scale, dropout):
        super(GNNModel, self).__init__()

        # order2
        self.conv2_1 = CustomGNNLayer2(in_features, args.hidden_features, args.alpha, args.beta, args.rho, args.scale)
        self.conv2_2 = CustomGNNLayer2(args.hidden_features, num_classes, args.alpha, args.beta, args.rho, args.scale)

        # order3
        self.conv3_1 = CustomGNNLayer3(in_features, args.hidden_features, args.rho, args.alpha, args.beta, args.scale)
        self.conv3_2 = CustomGNNLayer3(args.hidden_features, num_classes, args.rho, args.alpha, args.beta, args.scale)

        # order3
        self.conv4_1 = CustomGNNLayer4(in_features, args.hidden_features, args.rho, args.alpha, args.beta, args.scale)
        self.conv4_2 = CustomGNNLayer4(args.hidden_features, num_classes, args.rho, args.alpha, args.beta, args.scale)

        # attention
        self.multihead_attn = MultiheadAttention(embed_dim=num_classes, num_heads=num_heads, dropout=args.dropout)

        # FC
        self.fc = nn.Linear(in_features, args.hidden_features)

    
    def forward(self, x, adjs):
        # Min-Max normalization
        epsilon = args.epsilon
        x = (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + epsilon)

        # ----------------order 2
       
        # Apply the first GNN layer
        out1 = self.conv2_1(x, x, adjs)
        out1 = (out1 - out1.min(dim=1, keepdim=True)[0]) / (out1.max(dim=1, keepdim=True)[0] - out1.min(dim=1, keepdim=True)[0])
        out1 = torch.relu(out1)
        
        X_reduced = self.fc(x)
        X_reduced = F.dropout(X_reduced, training=self.training)
        # Apply the second GNN layer
        out1 = self.conv2_2(out1, X_reduced, adjs)
        out1 = F.log_softmax(out1, dim=1)


        # ----------------order 3
        out2 = self.conv3_1(x, x)
        out2 = (out2 - out2.min(dim=1, keepdim=True)[0]) / (out2.max(dim=1, keepdim=True)[0] - out2.min(dim=1, keepdim=True)[0])
        out2 = F.relu(out2)
        out2 = F.dropout(out2, training=self.training)
        out2 = self.conv3_2(out2, X_reduced)
        out2 = F.log_softmax(out2, dim=1)

        # ----------------order 4
        out3 = self.conv4_1(x, x)
        out3 = (out3 - out3.min(dim=1, keepdim=True)[0]) / (out3.max(dim=1, keepdim=True)[0] - out3.min(dim=1, keepdim=True)[0])
        out3 = F.relu(out3)
        out3 = F.dropout(out3, training=self.training)
        out3 = self.conv4_2(out3, X_reduced)
        out3 = F.log_softmax(out3, dim=1)
        
        

        gnn_outputs = torch.stack([out1, out2, out3], dim=0)  # Shape: [3, batch_size, num_classes]
        
        # Multi-head attention
        attn_output, _ = self.multihead_attn(gnn_outputs, gnn_outputs, gnn_outputs)
        attn_output = attn_output.mean(dim=0)

        out = F.log_softmax(attn_output, dim=1)
        del attn_output, out1, out2, out3,x, X_reduced
        gc.collect()  
        torch.cuda.empty_cache()  
        return out


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        self.val_loss_min = val_loss
