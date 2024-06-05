# -*- coding: utf-8 -*-
"""
Created on Tue May 28 01:19:34 2024

@author: Otman CH
"""


import torch
import torch.nn as nn
import torch.optim as optim
from util import *
import time

class FFN_1(nn.Module):
    def __init__(self, model_params):
        super(FFN_1, self).__init__()
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
       # init.normal_(self.fc2.bias, mean=0.0, std=0.1)
       # init.uniform_(self.fc3.bias, a=0.0, b=1.0)


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





    def train_model(self, device , groupe,seed,model, model_params, train_data, valid_data, test_set = None,
                      printFreq=100, max_patience = 100):

        
        self.best_val_loss = float('inf')
        patience=0
        optimizer = optim.AdamW(model.parameters(), lr=model_params['learning_rate'], betas=(0.9, 0.99),weight_decay=0.0001) ####0.001#############
        #pred_train = []
        #pred_val = []
        
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
    
                #pred_train.append(R_pred.detach().numpy())
        
                loss = self.criterion(Ytr_pred, Ytrain[mask_tr])
        
                train_epoch_loss = loss.item()
               
             
           
            #train_epoch_loss = total_loss #/ total_samples
            
            model.eval()
            #total_loss = 0
            #total_samples = 0
            with torch.no_grad():
    
                
                Y_pred_val = model(X_valid_m)
    
                #pred_val.append(R_pred.detach().numpy())
                valid_epoch_loss = self.criterion(Y_pred_val, Y_valid[mask_val])
                #valid_epoch_loss = loss.item()
                #total_samples += 1  
           
            if test_set :
                with torch.no_grad():    
                   Y_pred_test = model(Xtest_m)
        
                   sharp_test = evaluate_sharp(Y_pred_test.detach().cpu().numpy(),Ytest,mask_test)
                   test_epoch_loss = self.criterion(Y_pred_test, Ytest[mask_test])
                   r2_test = R2(Ytest[mask_test], Ytest[mask_test]- Y_pred_test.detach().cpu().numpy(),cross_sectional=False)
            
            sharp_train = evaluate_sharp(Ytr_pred.detach().cpu().numpy(),YT,mask_tr)
            sharp_valid = evaluate_sharp(Y_pred_val.detach().cpu().numpy(), YV, mask_val)
            r2_train = R2(YT[mask_tr], YT[mask_tr]- Ytr_pred.detach().cpu().numpy(),cross_sectional=False)
            r2_val = R2(YV[mask_val], YV[mask_val] - Y_pred_val.detach().cpu().numpy(),cross_sectional=False)
            #alid_epoch_loss = total_loss# / total_samples
            
            
                
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
    
    def compute_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss
    
    

    
