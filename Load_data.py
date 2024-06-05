# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:45:49 2024

@author: Otman CH

This class is used to construct training, validation and test datasets.


It takes the the paths of train, valid and test compressednpz data, it takes also the path of regime 
probabilities as well as the groupe of macro data used to construct thos probablities.
If goupe = None no regime probabilities will be use only characterestics will be used in the model.

First a mask is constructed to ignore unavailabel data before passing it to the 
model ( datamarked as -99), the mask is applied on characteristics as well as on macro probabilities

if macro is used: the data will be constructed by concatenating charactersitics, probabilites 
and the product characteristics*probabilites in each month: 
    data = [char[mask], p[mask]*char[mask], p[mask]]
if not : data = char[mask]

this is applied for train, validation and test.

The class has getters to retreive data (x,y,mask) for each dataset, and a get to retrieve 
the characteristics names.

"""


import numpy as np
import pickle

class Load_data():

    def __init__(self,train_tmp_path, test_tmp_path, valid_tmp_path, macro_paths=None, groupe=None):
        # macro paths macro to train_val and test paths in this order
            
        train_tmp = np.load(train_tmp_path)
        test_tmp = np.load(test_tmp_path)
        val_tmp = np.load(valid_tmp_path)
        data_train = train_tmp['data']
        data_test = test_tmp['data']
        data_val = val_tmp['data']
        
        # Divide into inputs and outputs
        Ytrain = data_train[:, :, 0]
        Ytest = data_test[:, :, 0]
        Yval = data_val[:, :, 0]
        Xtrain_Ch = data_train[:, :, 1:]
        Xtest_Ch = data_test[:, :, 1:]
        Xval_Ch = data_val[:, :, 1:]
        

        self.var_names = test_tmp['variable'][1:]
        
        # Create masks for the training, validation and test data
        mask_train = (Ytrain != -99.99)
        mask_test = (Ytest != -99.99)
        mask_valid = (Yval != -99.99)
        
        
        if groupe is  None:
            Xtrain_m = Xtrain_Ch[mask_train]
            X_val_m= Xval_Ch[mask_valid]
            Xtest_m = Xtest_Ch[mask_test]
            
            self.data_train= (Xtrain_m, Ytrain, mask_train)
            self.data_valid = (X_val_m, Yval, mask_valid)
            self.data_test = (Xtest_m, Ytest, mask_test)
        
       
        
        else:
            
            with open(macro_paths[0], 'rb') as file:
                prob_tr_val_macro = pickle.load(file)
                
            with open(macro_paths[1], 'rb') as file:
                prob_test_macro = pickle.load(file)
            
            # train and val have same prob file since hamilton filter was applied on all data
            prob_tr_val=prob_tr_val_macro[groupe]
            
            prob_train = prob_tr_val[:Xtrain_Ch.shape[0]].reshape(-1,1)
            prob_val = prob_tr_val[Xtrain_Ch.shape[0]:].reshape(-1,1)
            
            prob_test = prob_test_macro[groupe].reshape(-1,1)


            NSize = Ytrain.shape[1]
            macro_train = prob_train.repeat(NSize, 1) # make a proba for each asset
            macro_masked_tr = macro_train[mask_train].reshape(-1,1) # mask for unavailable assets char            
            char_masked = Xtrain_Ch[mask_train]
            
            char_mac_m = char_masked * macro_masked_tr # macro prob char combination

            Xtrain_m = np.concatenate((char_masked,char_mac_m, macro_masked_tr), axis=1)
            
            
            NSize = Ytest.shape[1]
            I_macro_test = prob_test.repeat(NSize, 1)
            I_macro_masked_test = I_macro_test[mask_test].reshape(-1,1)
            I_m_test = Xtest_Ch[mask_test]
            # multiply
            I_I_test = I_m_test * I_macro_masked_test
            Xtest_m = np.concatenate((I_m_test, I_I_test,I_macro_masked_test), axis=1)
            del NSize,I_macro_test,I_macro_masked_test,I_m_test

            NSize = Yval.shape[1]
            I_macro_val = prob_val.repeat(NSize, 1)
            I_macro_masked_val = I_macro_val[mask_valid].reshape(-1,1)
            I_masked_val= Xval_Ch[mask_valid]
            # multiply
            I_I_val = I_masked_val * I_macro_masked_val
            Xval_m = np.concatenate((I_masked_val,I_I_val, I_macro_masked_val), axis=1)
            del NSize,I_macro_val,I_macro_masked_val,I_masked_val


            self.data_train= (Xtrain_m, Ytrain, mask_train)
            self.data_valid = (Xval_m, Yval, mask_valid)
            self.data_test = (Xtest_m, Ytest, mask_test)
            
    def get_train(self):
        return self.data_train
    def get_val(self):
        return self.data_valid
    def get_test(self):
        return self.data_test
    def get_var_names(self):
        return self.var_names
        
            
        
 

        
        
