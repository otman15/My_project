# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:32:42 2024

@author: Otman CH
"""


import torch
import pickle
import gc
#from util import evaluate_sharp,long_short_portfolio,low_high_portfolio,calculateStatisticsDecile3,construct_decile_portfolios
from the_model import FFN_1
from HamiltonPCA import run_Hamilton_filter
#from Plotting import long_and_short,plot_L_S_portf_ret,plot_dec_port_basedOn_pred_ret
from Load_data import Load_data
import os


def run_model(seed,char_paths, macro_prob_paths,groupe, device, save=False):    
    
    train_tmp_path,valid_tmp_path,test_tmp_path = char_paths
    
    data = Load_data(train_tmp_path, test_tmp_path, valid_tmp_path,macro_prob_paths, groupe)
    
    data_train= data.get_train()
    data_valid = data.get_val()
    data_test = data.get_test()
    
    #test_variables = data.get_var_names()
    
    
    Xtest_m, Ytest, mask_test = data_test
    
    model_params = {'feature_dim':Xtest_m.shape[-1],
                    "num_epochs": 2000,'dropout':0.5,                                
                    "learning_rate": 0.001}
    model = FFN_1(model_params).to(device)
    
    best_params = None
    
    
    ######################################"
    model.train_model(device,groupe,seed,model, model_params, data_train, data_valid,
                  printFreq=20, max_patience=300)
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
        
    return pred



def run_all(save=False): 
    # Check if the probabilities file exists
    if not os.path.exists('macro_probabilities'):
        run_Hamilton_filter()
    else:
        print("Probabilities file already exists, execute run_Hamilton_filter() if you want to get and write prob again")
    
    
    
    groupes = [None ,'all_vars', 'all_groups', 'money_credit','Labor_market', 'Output_Income',  'Housing',
               'Consumption_OR',  'interest_exchange_rates', 'Prices', 'Stock_markets', 'Other']
    
    
    char_paths =("datasets/char/Char_train.npz", "datasets/char/Char_valid.npz" , "datasets/char/Char_test.npz")
    macro_prob_paths = ('macro_probabilities/macro_tr_prob.pkl','macro_probabilities/macro_test_prob.pkl')   
    
    seed = 991585  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    predictions = {}
    for groupe in groupes:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        res = run_model(seed,char_paths, macro_prob_paths,groupe, device, save=False)
        predictions[groupe] = res
        del res
        gc.collect()  # Explicitly call garbage collector
        torch.cuda.empty_cache()  # Free up GPU memory if used


    return predictions


predictions = run_all(save=False)











