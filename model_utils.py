# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:13:50 2024

@author: oemn
"""
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy, auroc, precision, specificity
from tqdm import tqdm
import gc
import os



def load_dataset(dataset_name):
    if dataset_name == 'Cora':
        return Planetoid(root='data/Cora', name='Cora')
    elif dataset_name == 'PubMed':
        return Planetoid(root='data/PubMed', name='PubMed')
    elif dataset_name == 'CiteSeer':
        return Planetoid(root='data/CiteSeer', name='CiteSeer')
    else:
        print(f'Unknown dataset: {dataset_name}')
        exit(1)


def create_loader(edge_index, node_idx, sizes, batch_size, shuffle, num_workers=0):
    return NeighborSampler(edge_index, node_idx=node_idx, sizes=sizes, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



def run_model(model, data, loader, device, is_train=False, optimizer=None):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    targeter = []                      
    preds = []                        
    for _, n_id, adjs in tqdm(loader):
        X = data.x[n_id].to(device)  # Move data to the device
        Y = data.y[n_id].to(device)  # Move data to the device
        edge_index = adjs[0].edge_index.to(device)
        adj_matrix_sparse = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(edge_index.shape[1], device=device), size=(n_id.size(0), n_id.size(0))).to(device)
        # Convert the sparse adjacency matrix to a dense matrix
        adj_matrix_sparse = adj_matrix_sparse.to_dense()
        if is_train:
            optimizer.zero_grad()
        out = model(X, adj_matrix_sparse)  # Forward propagation
        loss = F.nll_loss(out, Y)  # Compute the loss
        if is_train:
            loss.backward()  # Backward propagation
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update parameters
        total_loss += loss.item()
        preds.append(out.detach().cpu())   # Move to CPU before appending
        targeter.append(Y.detach().cpu())  # Move to CPU before appending
        gc.collect()  
        torch.cuda.empty_cache()  

    targeter = torch.cat(targeter,dim=0).cpu()
    preds = torch.cat(preds,dim=0).cpu()
    avg_loss = total_loss / len(loader)
    ACC = float('{:.6f}'.format(accuracy(preds,targeter, task="multiclass", num_classes=len(data.y.unique()))))
    AUC = float('{:.6f}'.format(auroc(preds,targeter, task="multiclass", num_classes=len(data.y.unique()))))
    Prec = float('{:.6f}'.format(precision(preds,targeter, task="multiclass", num_classes=len(data.y.unique()))))
    Spec = float('{:.6f}'.format(specificity(preds,targeter, task="multiclass", num_classes=len(data.y.unique()))))
    if is_train:
        return model, avg_loss, ACC, AUC, Prec, Spec
    else:
        return avg_loss, ACC, AUC, Prec, Spec




def train(model, data, train_loader, optimizer, device):
    return run_model(model, data, train_loader, device, is_train=True, optimizer=optimizer)

def validate(model, data, val_loader, device):
    return run_model(model, data, val_loader, device)

def test(model, data, test_loader, device):
    return run_model(model, data, test_loader, device)



def save_model_path(directory, filename):
    os.makedirs(directory, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(filename))[0]
    model_path = f"{directory}/{script_name}_model.pth"
    return model_path