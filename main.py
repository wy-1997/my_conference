# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:19:29 2024

@author: oemn
"""
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from arg_parser import args
from gnn_model import GNNModel, EarlyStopping
from model_utils import load_dataset, create_loader, run_model, train,validate,test,save_model_path



# load dataset
dataset = load_dataset(args.dataset)
data = dataset[0]
#NeighborSampler
train_loader = create_loader(data.edge_index, data.train_mask, args.sizes, args.batch_size, True)
val_loader = create_loader(data.edge_index, data.val_mask, args.sizes, args.batch_size, False)
test_loader = create_loader(data.edge_index, data.test_mask, args.sizes, args.batch_size, False)


# Check GPUs
device_ids = args.device_ids  # Change this to the IDs of the GPUs you want to use
if torch.cuda.is_available() and max(device_ids) < torch.cuda.device_count():
    print(f"Let's use GPUs {device_ids}!")
    model = nn.DataParallel(GNNModel(data.num_features, args.hidden_features, dataset.num_classes, dataset.num_classes, args.alpha, args.beta, args.rho, args.scale, args.dropout), device_ids=device_ids)
else:
    print("Let's use single GPU or CPU!")
    model = GNNModel(data.num_features, args.hidden_features, dataset.num_classes, dataset.num_classes, args.alpha, args.beta, args.rho, args.scale, args.dropout)

device = torch.device(f'cuda:{device_ids[0]}')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


# Specify the directory where you want to save the log file
log_dir = args.log_dir # relative path to the log directory


# Check if the directory exists
if not os.path.exists(log_dir):
    print(f"Directory {log_dir} does not exist. Creating it.")
    os.makedirs(log_dir, exist_ok=True)
else:
    print(f"Directory {log_dir} already exists.")



# Set up logging
log_filename = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + os.path.basename(__file__) + '.log')
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 
early_stopping = EarlyStopping(patience=args.patience, verbose=True)
num_epochs = args.epochs

start_time = time.time()
for epoch in range(num_epochs):
    MODEL, train_avg_loss, train_ACC, train_AUC, train_Prec, train_Spec = train(model, data, train_loader, optimizer, device)
    end_time = time.time()
    
    # training
    training_duration = end_time - start_time
    print(f'Epoch {epoch+1}, Train_Loss: {train_avg_loss:.4f}, Train_Acc: {train_ACC:.4f}, Train_Auc: {train_AUC:.4f}, Train_Prec: {train_Prec:.4f}, Train_Spec: {train_Spec:.4f}')
    
    # validate
    val_avg_loss, val_ACC, val_AUC, val_Prec, val_Spec = validate(model, data, val_loader, device)
    end_time_val = time.time()
    val_duration = end_time_val - start_time
    
     # Log the results
    logging.info(f'Epoch {epoch+1}, Train_Loss: {train_avg_loss:.4f}, Train_Acc: {train_ACC:.4f}, Train_Auc: {train_AUC:.4f}, Train_Prec: {train_Prec:.4f}, Train_Spec: {train_Spec:.4f},Training duration: {training_duration:.2f} seconds\
                 Validation, Val_Loss: {val_avg_loss:.4f}, Val_Acc: {val_ACC:.4f}, Val_Auc: {val_AUC:.4f}, Val_Prec: {val_Prec:.4f}, Val_Spec: {val_Spec:.4f}, Validation duration: {val_duration:.2f} seconds')
    # example
    model_path = save_model_path('./model', __file__)
    # check earlystop
    early_stopping(val_avg_loss, MODEL)
    if early_stopping.early_stop:
        print("Early stopping")
        torch.save(MODEL.state_dict(), model_path)
        break


test_avg_loss, test_ACC, test_AUC, test_Prec, test_Spec = test(model, data, test_loader, device)
print(f'Test, Test_Loss: {test_avg_loss:.4f}, Test_Acc: {test_ACC:.4f}, Test_Auc: {test_AUC:.4f}, Test_Prec: {test_Prec:.4f}, Test_Spec: {test_Spec:.4f}')
end_time_test = time.time()
test_duration = end_time_test - start_time
logging.info(f'Test, Test_Loss: {test_avg_loss:.4f}, Test_Acc: {test_ACC:.4f}, Test_Auc: {test_AUC:.4f}, Test_Prec: {test_Prec:.4f}, Test_Spec: {test_Spec:.4f}, Test duration: {test_duration:.2f} seconds')
