# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:45:49 2024

@author: Otman CH

  Cette classe est utilisée pour construire des ensembles de données d'entraînement,
de validation et de test.

  Elle prend en entrée les chemins des données compressées npz pour l'entraînement,
la validation et le test, ainsi que le chemin des probabilités de régime 
et le groupe de données macro utilisé pour construire ces probabilités.
Si `groupe = None`, aucune probabilité de régime ne sera utilisée et seules
les caractéristiques seront utilisées dans le modèle.

 Tout d'abord, un masque est construit pour ignorer les caractéristiques non disponibles
avant de les passer au modèle (données marquées comme -99).
Ce masque est appliqué aux caractéristiques ainsi qu'aux probabilités macro 
si elles sont utilisées.

 Si les données macro sont utilisées, les données seront construites en concaténant
les caractéristiques, les probabilités et le produit caractéristiques*probabilités
pour chaque mois et chaque actif :
data = [char[mask], p[mask]*char[mask], p[mask]]

Sinon, les données seront simplement les caractéristiques :
data = char[mask]

Cela est appliqué pour l'ensemble d'entraînement, de validation et de test.

 La classe dispose de méthodes pour récupérer les données (x, y, masque)
pour chaque ensemble de données, ainsi qu'une méthode pour récupérer
les noms des caractéristiques.

"""


import numpy as np
import pickle

class Load_data():

    def __init__(self,train_tmp_path, test_tmp_path, valid_tmp_path, macro_paths=None, groupe=None):
        # Chemins des données macro d'entraînement/validation et de test dans cet ordre
            
        train_tmp = np.load(train_tmp_path)
        test_tmp = np.load(test_tmp_path)
        val_tmp = np.load(valid_tmp_path)
        data_train = train_tmp['data']
        data_test = test_tmp['data']
        data_val = val_tmp['data']
        
        # Diviser en entrées et sorties
        Ytrain = data_train[:, :, 0]
        Ytest = data_test[:, :, 0]
        Yval = data_val[:, :, 0]
        Xtrain_Ch = data_train[:, :, 1:]
        Xtest_Ch = data_test[:, :, 1:]
        Xval_Ch = data_val[:, :, 1:]
        

        self.var_names = test_tmp['variable'][1:]
        
        # Créer des masques pour les données d'entraînement, de validation et de test
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
            
            # Les ensembles d'entraînement et de validation ont le même fichier de probabilités puisque le filtre de Hamilton a été appliqué sur les deux en même temps.
            prob_tr_val=prob_tr_val_macro[groupe]
            
            prob_train = prob_tr_val[:Xtrain_Ch.shape[0]].reshape(-1,1)
            prob_val = prob_tr_val[Xtrain_Ch.shape[0]:].reshape(-1,1)
            
            prob_test = prob_test_macro[groupe].reshape(-1,1)


            NSize = Ytrain.shape[1]
            macro_train = prob_train.repeat(NSize, 1) # Créer une probabilité pour chaque actif
            macro_masked_tr = macro_train[mask_train].reshape(-1,1) # Créer une probabilité pour chaque actif            
            char_masked = Xtrain_Ch[mask_train]
            
            char_mac_m = char_masked * macro_masked_tr # combinaison macro prob et char

            Xtrain_m = np.concatenate((char_masked,char_mac_m, macro_masked_tr), axis=1)
            
            
            NSize = Ytest.shape[1]
            I_macro_test = prob_test.repeat(NSize, 1)
            I_macro_masked_test = I_macro_test[mask_test].reshape(-1,1)
            I_m_test = Xtest_Ch[mask_test]
            
            I_I_test = I_m_test * I_macro_masked_test
            Xtest_m = np.concatenate((I_m_test, I_I_test,I_macro_masked_test), axis=1)
            del NSize,I_macro_test,I_macro_masked_test,I_m_test

            NSize = Yval.shape[1]
            I_macro_val = prob_val.repeat(NSize, 1)
            I_macro_masked_val = I_macro_val[mask_valid].reshape(-1,1)
            I_masked_val= Xval_Ch[mask_valid]
            
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
        
            
        
 

        
        
