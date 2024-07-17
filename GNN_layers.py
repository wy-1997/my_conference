# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:25:18 2024

@author: oemn
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:19:17 2024

@author: oemn
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import numpy as np



class CustomGNNLayer2(nn.Module):
    def __init__(self, in_features, out_features, rho, alpha, beta, scale):
        super(CustomGNNLayer2, self).__init__()
        # Initialize parameters
        self.in_features = in_features
        self.out_features = out_features
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.batch_norm = nn.BatchNorm1d(out_features)

        # Define the weight matrix for feature transformation
        self.W_mlp = nn.Linear(in_features, out_features)


    def reset_parameters(self):
        # Initialize the weight matrix W using Xavier initialization
        nn.init.kaiming_uniform_(self.W_mlp.weight, nonlinearity='relu')
        # Initialize the bias terms
        nn.init.zeros_(self.W_mlp.bias)



    def forward(self, H, X, A):
        """
        Apply the node representation update rule within a GNN layer.
        
        H: Node representations of shape (N, F), where N is the number of nodes and F is the number of features.
        X: Input features of shape (N, F).
        """

        N,_ = X.shape
        I = torch.eye(N, device=H.device)
        A = A@A.T
        term1 = (I - 2*self.rho*self.alpha*I - 2*self.beta*A)@self.W_mlp(H)      # H_transformed
        term2 = 2*self.rho*self.alpha * self.W_mlp(X)                         # X_transformed                                  
        out = term1 + term2
        del term1, term2, I
        out = self.batch_norm(out)
        gc.collect()  
        torch.cuda.empty_cache()  
        return out





class CustomGNNLayer4(nn.Module):
    def __init__(self, in_features, out_features, rho, alpha, beta, scale):
        super(CustomGNNLayer4, self).__init__()
        # Initialize parameters
        self.in_features = in_features
        self.out_features = out_features
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.batch_norm = nn.BatchNorm1d(out_features)

        # Define the weight matrix for feature transformation
        self.W_mlp = nn.Linear(in_features, out_features)
        self.W_alpha = nn.Linear(out_features, out_features)
        self.W_beta = nn.Linear(in_features, out_features)




    def reset_parameters(self):
        # Initialize the weight matrix W using Xavier initialization


        nn.init.kaiming_uniform_(self.W_mlp.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_alpha.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_beta.weight, nonlinearity='relu')

        # Initialize the bias terms
        nn.init.zeros_(self.W_mlp.bias)
        nn.init.zeros_(self.W_alpha.bias)
        nn.init.zeros_(self.W_beta.bias)


    def squared_distance(self, X):
        r = torch.sum(X * X, dim=1)
        r = r.view(-1, 1)
        D = r - 2 * torch.mm(X, X.t()) + r.view(1, -1)
        return D

    def full_affinity(self, X, scale):
        '''
        Calculates the symmetrized full Gaussian affinity matrix, scaled
        by a provided scale

        X: input dataset of size (n, d) where n is the number of points and d is the dimension
        scale: provided scale

        returns: n x n affinity matrix
        '''
        # Ensure scale is not too small to avoid division by zero
        scale = max(scale, 1e-7)
        sigma_squared = torch.tensor([scale ** 2], dtype=X.dtype, device=X.device)

        Dx = self.squared_distance(X)

        # Ensure Dx does not contain any negative values
        Dx = torch.clamp(Dx, min=0)

        Dx_scaled = Dx / (2 * sigma_squared + 1e-7)

        # Ensure Dx_scaled does not contain any extremely large values
        Dx_scaled = torch.clamp(Dx_scaled, max=100)

        W = torch.exp(-Dx_scaled).clamp(min=1e-7)

        # Add a small constant to avoid division by zero in softmax
        W_normalized = F.softmax(W + 1e-7, dim=1)  # 行归一化

        return W_normalized






    def forward(self, H, X):
        """
        Apply the node representation update rule within a GNN layer.
        
        H: Node representations of shape (N, F), where N is the number of nodes and F is the number of features.
        X: Input features of shape (N, F).
        """

        N, F = H.shape
        I = torch.eye(N, device=H.device)
        T4_affinity = self.full_affinity(H, self.scale)
        T_4 = torch.einsum('ij,mk->ijmk',T4_affinity,T4_affinity)
        T_4 = torch.reshape(T_4, (N*N, -1))
        jitter = 1e-6  # or any small positive number
        T_4 = T_4 + jitter * torch.eye(T_4.size(0)).to(T_4.device)


        T_4 = torch.nn.functional.normalize(T_4, dim=0)
        S, _ = torch.linalg.qr(T_4)
        B = torch.einsum('ik,jk->ijk', H, H).reshape(N*N, -1)

        del T_4, T4_affinity

        term2 = self.W_alpha(self.W_mlp(X))  # X_transformed                                  
        term1 = self.W_mlp(H)  @ (torch.eye(self.out_features).to(H.device) - self.W_alpha.weight)                    # H_transformed
        del X



        # skip connection for term1 and term2
        sum_part = 0

        for s in range(F):
            # Extract the s-th feature across all nodes
            H_s = H[:, s:s+1]
            sum_part += (torch.kron(I, H_s) + torch.kron(I, H_s)).T       # shape 

        sum_part = sum_part / torch.norm(sum_part) if torch.norm(sum_part) != 0 else sum_part        # norm
        S = S / torch.norm(S) if torch.norm(S) != 0 else S        # norm
        B = B / torch.norm(B) if torch.norm(B) != 0 else B        # norm
        term3 = sum_part @ S @ B      # shape (N, N^2)

        term3 = self.W_beta(term3)    
        out = term1 + term2  + term3
        out = self.batch_norm(out)

        del term1, term2, term3, S, B, I, sum_part, H_s,   # Delete the tensors to free memory
        gc.collect()  
        torch.cuda.empty_cache()  

        return out
    
class CustomGNNLayer3(nn.Module):
    def __init__(self, in_features, out_features, rho, alpha, beta, scale):
        super(CustomGNNLayer3, self).__init__()
        # Initialize parameters
        self.in_features = in_features
        self.out_features = out_features
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.batch_norm = nn.BatchNorm1d(out_features)

        # Define the weight matrix for feature transformation
        self.W_mlp = nn.Linear(in_features, out_features)





    def reset_parameters(self):
        # Initialize the weight matrix W using Xavier initialization
        nn.init.kaiming_uniform_(self.W_mlp.weight, nonlinearity='relu')

        # Initialize the bias terms
        nn.init.zeros_(self.W_mlp.bias)


    def squared_distance(self, X):
        r = torch.sum(X * X, dim=1)
        r = r.view(-1, 1)
        D = r - 2 * torch.mm(X, X.t()) + r.view(1, -1)
        return D

    def full_affinity(self, X, scale):
        '''
        Calculates the symmetrized full Gaussian affinity matrix, scaled
        by a provided scale

        X: input dataset of size (n, d) where n is the number of points and d is the dimension
        scale: provided scale

        returns: n x n affinity matrix
        '''
        # Ensure scale is not too small to avoid division by zero
        scale = max(scale, 1e-7)
        sigma_squared = torch.tensor([scale ** 2], dtype=X.dtype, device=X.device)

        Dx = self.squared_distance(X)

        # Ensure Dx does not contain any negative values
        Dx = torch.clamp(Dx, min=0)

        Dx_scaled = Dx / (2 * sigma_squared + 1e-7)

        # Ensure Dx_scaled does not contain any extremely large values
        Dx_scaled = torch.clamp(Dx_scaled, max=100)

        W = torch.exp(-Dx_scaled).clamp(min=1e-7)

        # Add a small constant to avoid division by zero in softmax
        W_normalized = F.softmax(W + 1e-7, dim=1)  # 行归一化

        return W_normalized





    def forward(self, H, X):
        """
        Apply the node representation update rule within a GNN layer.
        
        H: Node representations of shape (N, F), where N is the number of nodes and F is the number of features.
        X: Input features of shape (N, F).
        """

        N, F = H.shape
        I = torch.eye(N, device=H.device)
        T3_affinity = self.full_affinity(H, self.scale)
        T_3 = torch.einsum('ni,nj->nij', T3_affinity, T3_affinity)
        del T3_affinity
        
        term1 = (1- 2*self.rho*self.alpha)*self.W_mlp(H)                         # H_transformed
        term4 = 2*self.rho*self.alpha * self.W_mlp(X)                         # X_transformed                                  
        term5 = 0

        
        for s in range(F):
            # Extract the s-th feature across all nodes
            H_s = H[:, s:s+1]
            term2 = torch.matmul(torch.reshape(T_3, (N, N*N)), torch.kron(H_s, H_s))
            term3_part1 = torch.kron(I, H_s).T + torch.kron(H_s, I).T
            term3_part2 = torch.matmul(torch.reshape(T_3, (N, N*N)).T, H_s)
            term3 =  term3_part1 @ term3_part2
            term5 += self.rho * self.beta *(term2+term3)
        term5 = term5 / torch.norm(term5) if torch.norm(term5) != 0 else term5        # norm


        
        out = term1 + term4  + term5
        del term1, term4, term5, term3, term2, term3_part1, term3_part2
        out = self.batch_norm(out)

        gc.collect() 
        torch.cuda.empty_cache()  

        return out