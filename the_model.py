# -*- coding: utf-8 -*-
"""
Created on Tue May 28 01:19:34 2024

@author: Otman CH

This class implements a Feed-Forward Neural Network (FFN_1) using PyTorch for regression prediction tasks. 
It includes methods for defining the network architecture, forward propagation, and model training 
with early stopping based on validation loss. 
The network architecture consists of multiple fully connected layers with ReLU activation and dropout 
for regularization. The training method utilizes the AdamW optimizer and Mean Squared Error loss,
and tracks metrics such as Sharpe ratio and R^2 for train, validation, and test sets.


"""


import torch
import torch.nn as nn
import torch.optim as optim
from util import sharp, R2
import time



class FFN_1(nn.Module):
    def __init__(self, model_params):
        super(FFN_1, self).__init__()
        
        # Define the network architecture
        self._individual_feature_dim = model_params['feature_dim']
        self._dropout = model_params['dropout']
        self.model_params = model_params

        self.fc1 = nn.Linear(self._individual_feature_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self._dropout)
        self.criterion =  nn.MSELoss()
        self.best_params = None


    # Define the forward propagation
    def forward(self, x):


        r = self.relu(self.fc1(x))
        r = self.dropout(r)
        r = self.relu(self.fc2(r))
        r = self.dropout(r)
        r = self.relu(self.fc3(r))
        r = self.dropout(r)
        r = self.relu(self.fc4(r))
        r = self.dropout(r)
        r_pred = self.fc5(r)

        return r_pred.view(-1)




    # train the model for each group of data
    def train_model(self, device , groupe,seed,model, model_params, train_data, valid_data, test_set = None,
                      printFreq=100, max_patience = 100):
        
        # paramters initialization and data preparation for input layer
        
        self.best_val_loss = float('inf')
        patience=0
        optimizer = optim.AdamW(model.parameters(), lr=model_params['learning_rate'], betas=(0.9, 0.99),weight_decay=0.0001) ####0.001#############

        
        XV, YV, mask_val = valid_data
        XT, YT, mask_tr = train_data
        
        Xtrain_m = torch.tensor(XT, dtype=torch.float, requires_grad=True).to(device)
        Ytrain = torch.tensor(YT, dtype=torch.float, requires_grad=True).to(device)

        X_valid_m = torch.tensor(XV, dtype=torch.float).to(device)
        Y_valid = torch.tensor(YV, dtype=torch.float).to(device)

        if test_set :
            xs, ys, mask_test = test_set

            Xtest_m = torch.tensor(xs, dtype=torch.float).to(device)
            Ytest = torch.tensor(ys, dtype=torch.float).to(device)
        
    
        start_time = time.time()
        
        # training using forward backward propagation
        for epoch in range(model_params['num_epochs']):
                  
            model.train()

            Ytr_p = model(Xtrain_m)
            optimizer.zero_grad()
            loss = self.criterion(Ytr_p, Ytrain[mask_tr])

            loss.backward()
            optimizer.step()
            
      
            
            model.eval()
            with torch.no_grad():
                Ytr_pred = model(Xtrain_m)

        
                loss = self.criterion(Ytr_pred, Ytrain[mask_tr])
        
                train_epoch_loss = loss.item()
               

            
            model.eval()

            with torch.no_grad():
    
                
                Y_pred_val = model(X_valid_m)

                valid_epoch_loss = self.criterion(Y_pred_val, Y_valid[mask_val])

            if test_set :
                with torch.no_grad():    
                   Y_pred_test = model(Xtest_m)
        
                   sharp_test = sharp(Y_pred_test.detach().cpu().numpy(),Ytest,mask_test)
                   test_epoch_loss = self.criterion(Y_pred_test, Ytest[mask_test])
                   r2_test = R2(Ytest[mask_test], Ytest[mask_test]- Y_pred_test.detach().cpu().numpy(),cross_sectional=False)
            
            sharp_train = sharp(Ytr_pred.detach().cpu().numpy(),YT,mask_tr)
            sharp_valid = sharp(Y_pred_val.detach().cpu().numpy(), YV, mask_val)
            r2_train = R2(YT[mask_tr], YT[mask_tr]- Ytr_pred.detach().cpu().numpy(),cross_sectional=False)
            r2_val = R2(YV[mask_val], YV[mask_val] - Y_pred_val.detach().cpu().numpy(),cross_sectional=False)

            
            
                
            if valid_epoch_loss < self.best_val_loss:
                self.best_val_loss = valid_epoch_loss
                self.best_params = model.state_dict()
                patience = 0
            else:
                    patience += 1
            if patience > max_patience: ## early stop when patience become larger than 40
                print('patience > ', max_patience)
                break
        
    
                
            if epoch >= printFreq and epoch % printFreq == 0:
    
                
                print('\n\n')
                
                
                print('macro_group: %s  Epoch n° :  %d  seed : %d' %(groupe, epoch, seed))
                         
                if test_set :
                    print('Epoch %d train/valid/test loss: %0.4f/%0.4f/%0.4f' %(epoch,train_epoch_loss, valid_epoch_loss,test_epoch_loss))
                    print('Epoch %d train/valid/test sharp: %0.4f/%0.4f/%0.4f' %(epoch, sharp_train, sharp_valid,sharp_test))
                    print('Epoch %d train/valid/test R2: %0.4f/%0.4f/%0.4f' %(epoch, r2_train, r2_val, r2_test))
                else:
                    print('Epoch %d train2/valid loss: %0.4f/%0.4f' %(epoch,train_epoch_loss, valid_epoch_loss))
                    print('Epoch %d train/valid sharp: %0.4f/%0.4f' %(epoch, sharp_train, sharp_valid))
                    print('Epoch %d train/valid R2: %0.4f/%0.4f' %(epoch, r2_train, r2_val))     
                    
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f" {printFreq } Epochs completed in {elapsed_time:.2f} seconds")
                start_time = time.time()
    

    def get_best_params(self):
        return self.best_params
    
    def get_best_val_loss(self):
        return self.best_val_loss


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN_1(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(FFN_1, self).__init__()
        
        # Define the network architecture
        self.hidden_layers = nn.ModuleList()
        last_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        
        self.output_layer = nn.Linear(last_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    # Define the forward propagation
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
'''
    
