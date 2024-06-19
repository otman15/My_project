# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:32:42 2024

@author: Otman CH

Ici les fonctions pour lancer le modèle.
"""


import torch
import pickle
import gc

from the_model import FFN_1
from HamiltonPCA import run_Hamilton_filter

from Load_data import Load_data
import os
import time


'''
  La fonction run_model prépare les données, initialise les paramètres puis lance l'entrainement 
pour un groupe donnée.
  Les résulats d'évaluation sur l'ensemble 'test_set' et les meilleurs paramèters sont écrit sur
un fichier pickle.
'''

def run_model(seed,char_paths, macro_prob_paths,groupe, device, save=False):    
    
    train_tmp_path,valid_tmp_path,test_tmp_path = char_paths
    
    data = Load_data(train_tmp_path, test_tmp_path, valid_tmp_path,macro_prob_paths, groupe)
    
    data_train= data.get_train()
    data_valid = data.get_val()
    data_test = data.get_test()
       
    Xtest_m, Ytest, mask_test = data_test
    
    model_params = {'feature_dim':Xtest_m.shape[-1],
                    "num_epochs": 2000,'dropout':0.5,                                
                    "learning_rate": 0.001,
                    'layer_sizes': [64, 32, 16, 8]}
    model = FFN_1(model_params).to(device)
    
    best_params = None
       
    ############################################
    model.train_model(device,groupe,seed,model, model_params, data_train, data_valid,
                  printFreq=50, max_patience=300)
    ############################################
    
    best_params = model.get_best_params()
    model.eval()
    model.load_state_dict(best_params)
    
        
    if save:
      if best_params is not None:
           torch.save(best_params, f'final_results/best_params/{groupe}_best_params.pth')
    
    
    with torch.no_grad():    
       Y_pred_test = model(torch.tensor(Xtest_m, dtype=torch.float).to(device))
    pred = Y_pred_test.detach().cpu().numpy()

    if save : 
       with open(f'final_results/results/{groupe}_pred.pkl', 'wb') as f:
           pickle.dump(pred, f)
    del model
        

'''
Cette fonction execute run_model itérativement sur tous les groupe de données.
Permet d'utiliser le gpu si disponible.
Des pauses d'execution sont introduite entre les itérations pour soulager les processeurs.
'''


def run_all(save=False, Pause = True): 
      
    #On utilise les prob sans lag
    macro_prob_paths = ('macro_probabilities/macro_tr_prob_sans_retard.pkl','macro_probabilities/macro_test_prob_sans_retard.pkl') 
    
    # Vérifier si les probabilités de régimes sont déja sur le disque 
    if not (os.path.exists(macro_prob_paths[0]) and os.path.exists(macro_prob_paths[1])):
        print("running Hamilton filter")
        run_Hamilton_filter()
    else:
        print("Probabilities file already exists, execute run_Hamilton_filter() if you want to get and write prob again")
       
    
    groupes = ['Production_Revenus', 'Marche_de_travail', 'Logement','Conso_Ordres' , 
               'Monnaie_credit', 'Taux_interet_change', 'Prix', 'Marches_Boursiers', 'Autres',
               'Toutes_vars', None,'Tous_groupes']
    
    
    char_paths =("datasets/char/Char_train.npz", "datasets/char/Char_valid.npz" , "datasets/char/Char_test.npz")
      
       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    pause_duration = 10
    i = 1
    for groupe in groupes:
        print('\n')
        print("groupe:",i)
        start_time = time.time()
        seed = 991585 
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        run_model(seed,char_paths, macro_prob_paths,groupe, device, save=save)
       
        elapsed_time = time.time() - start_time
        print("group training time = " ,elapsed_time)
        i=i+1
        gc.collect()  # Appeler garbage collector explicitement 
        torch.cuda.empty_cache()  
        
        # Une pause optionelle entre les itérations 
        if (Pause and i<11):
            pause_duration = pause_duration + 30  
            print(f"Pausing execution for {pause_duration} seconds...")
            time.sleep(pause_duration)       
            print("Resuming execution after pause")




run_all(save=True)











