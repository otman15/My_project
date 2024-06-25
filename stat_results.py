
   

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:32:53 2024

@author: Otman CH
"""

import pickle
from util import sharp, decile_portfolios,long_short_portfolio,DecileStatistics, CS_R2_decile,R2,calculateXSR2
from Load_data import Load_data
import pandas as pd
import matplotlib.pyplot as plt
from Plotting import plot_multi_dec_port_basedOn_pred_ret,plot_multi_L_S_portf_ret,plot_macro_probs
from the_model import FFN_1
import torch



############################################proba Regime #####################################################################""    
      
plot_macro_probs([('macro_probabilities/macro_tr_prob_avec_retard.pkl', 'macro_probabilities/macro_test_prob_avec_retard.pkl')], save=True, lag= 'avec autoregression')
plot_macro_probs([('macro_probabilities/macro_tr_prob_sans_retard.pkl', 'macro_probabilities/macro_test_prob_sans_retard.pkl')], save=True, lag= 'sans autoregression')  







################# Obtenir les données de test et les prédictions, puis effectuez les calculs nécessaires. ###################################"
train_tmp_path = "datasets/char/Char_train.npz"
test_tmp_path  = "datasets/char/Char_test.npz"
valid_tmp_path = "datasets/char/Char_valid.npz" 

macro_paths = ('macro_probabilities/macro_tr_prob_sans_retard.pkl','macro_probabilities/macro_test_prob_sans_retard.pkl') 
  
groupes = ['Toutes_vars', 'Tous_groupes','Marche_de_travail', 'Production_Revenus',  'Logement',
           'Conso_Ordres',  'Taux_interet_change', 'Marches_Boursiers', 'Prix', 'Monnaie_credit', 'Autres', None]

   


decile_portf_on_ret = {} # dict pour le rendement des portfeuilles de déciles triés par pred de rendt, pour chaque groupe macro a part
decile_portf_on_ret_pred = {} # dict pour la prediction du rendement ....
ls_returns = {} # dict pour le rendement du potefuille long court par groupe

CSR2_mom = {}  # pour le CS R2 des rendements des portfeuilles triés par moment
CSR2_size = {} # .................................................par size
momentum_sorting = 'r12_2' # Pour trier les portefeuilles en se basant sur le moment
size_sorting = 'LME' # Pour trier les portefeuilles en se basant sur le moment

sharps_test = {} # pour le sharp des portefuille long court 
sharps_val = {}   # pour le sharp des portefuille long court
sharps_train = {}  # pour le sharp des portefuille long court

CS_R2_tr = {} # CR R2 des predictions de rendements des actif entrainement
CS_R2_test = {} # CR R2 des predictions de rendements des actif test
CS_R2_val = {} # CR R2 des predictions de rendements des actif validation




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



for groupe in groupes:
    
      data = Load_data(train_tmp_path, test_tmp_path, valid_tmp_path,macro_paths, groupe)
      Xtest, Ytest, mask_test = data.get_test()
      Xtrain, Ytrain, mask_train = data.get_train()
      Xval, Yval, mask_val = data.get_val()     
      char_names = data.get_var_names()
      
      
      model_params = {'feature_dim':Xtest.shape[-1],
                      "num_epochs": 2000,'dropout':0.5,                                
                      "learning_rate": 0.001,
                      'layer_sizes': [64, 32, 16, 8]}
      model = FFN_1(model_params).to(device)      
      model.load_state_dict(torch.load(f'final_results/best_params/{groupe}_best_params.pth'))
      model.eval()
      
      
      
      pred_train = model(torch.tensor(Xtrain, dtype=torch.float).to(device)).detach().cpu().numpy()
      pred_val = model(torch.tensor(Xval, dtype=torch.float).to(device)).detach().cpu().numpy()
      pred_test = model(torch.tensor(Xtest, dtype=torch.float).to(device)).detach().cpu().numpy()
      
            
            
      sharps_test[groupe] = sharp(pred_test,Ytest,mask_test)
      sharps_train[groupe] = sharp(pred_train,Ytrain,mask_train)
      sharps_val[groupe] = sharp(pred_val,Yval,mask_val)
      
      
      
      CS_R2_tr[groupe]   = calculateXSR2(pred_train,  Ytrain, mask_train)
      CS_R2_val[groupe]  = calculateXSR2(pred_val,  Yval, mask_val)
      CS_R2_test[groupe] = calculateXSR2(pred_test,  Ytest, mask_test)
      
      
      decile_portf_on_ret[groupe],decile_portf_on_ret_pred[groupe] = decile_portfolios(pred_test, Ytest, mask_test, deciles=10)
      ls_returns[groupe] = long_short_portfolio(pred_test, Ytest, mask_test, low=0.1, high=0.1, normalize=True)
    
      var2idx = {var:idx for idx, var in enumerate(char_names)}
      char_mask_test_mom = Xtest[:,var2idx[momentum_sorting]]
      char_mask_test_size = Xtest[:,var2idx[size_sorting]]
      
      CSR2_mom[groupe] = DecileStatistics(char_mask_test_mom, Ytest, pred_test, mask_test, decile=10)
      CSR2_size[groupe] = DecileStatistics(char_mask_test_size, Ytest, pred_test, mask_test, decile=10)
      
    
      del data, Xtest, mask_test, Ytest, pred_test
      
      
###################################### portf decile  basé sur pred de rend ############################################################

potf_ret_list = list(decile_portf_on_ret.values())
macro_group_list = list(decile_portf_on_ret.keys())

plot_multi_dec_port_basedOn_pred_ret(potf_ret_list, macro_group_list, save=True)






#########################################portf LS  basé sur pred de rend ###########################################################

ls_returns_list = list(ls_returns.values())
macro_group_list = list(ls_returns.keys())

plot_multi_L_S_portf_ret(ls_returns_list, macro_group_list, save=True)





############################### XS R2 #############################""

if None in CS_R2_test:
    CS_R2_test['Sans macro'] = CS_R2_test.pop(None)
    CS_R2_tr['Sans macro']   = CS_R2_tr.pop(None)
    CS_R2_val['Sans macro']  = CS_R2_val.pop(None)


# Create a DataFrame from the dictionary

dfr = pd.DataFrame({
    'groupe macro': CS_R2_tr.keys(),
    'CS_R2_train': CS_R2_tr.values(),
    'CS_R2_val': CS_R2_val.values(),
    'CS_R2_test': CS_R2_test.values()
})

dfr = dfr.sort_values(by='CS_R2_test',ascending=False)
dfr = dfr.round(decimals=3)

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=dfr.values, colLabels=dfr.columns, cellLoc='center', loc='center')

# Adjust layout to make room for the title
plt.subplots_adjust(top=0.9)

# Add title to the table
#plt.suptitle('Ratio de sharpe OOS L_C portf', fontsize=12, y=0.85)

plt.savefig('final_results/figs/Table_R2_CS.jpg', bbox_inches='tight')




############################### ratio de Sharpe  LS portf basé sur pred #############################""

if None in sharps_test:
    sharps_test['Sans macro'] = sharps_test.pop(None)
    sharps_train['Sans macro'] = sharps_train.pop(None)
    sharps_val['Sans macro'] = sharps_val.pop(None)


# Create a DataFrame from the dictionary

df = pd.DataFrame({
    'groupe macro': sharps_test.keys(),
    'sharp_train': sharps_train.values(),
    'sharp_val': sharps_val.values(),
    'sharp_test': sharps_test.values()
})

df = df.sort_values(by='sharp_test',ascending=False)
df = df.round(decimals=3)

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Adjust layout to make room for the title
plt.subplots_adjust(top=0.9)

# Add title to the table
#plt.suptitle('Ratio de sharpe OOS L_C portf', fontsize=12, y=0.85)

plt.savefig('final_results/figs/Table_ratios_de_sharpe.jpg', bbox_inches='tight')










###################################### stat portefeuilles de déciles triés par moment  ##########################"
############## Just pour abrévier les noms.


R2_CS_m = {}
R2_CS_decile_m={}

for group in CSR2_mom.keys(): #  r2_cs et r2_cs_decile
  R2_CS_m[group] =CSR2_mom[group][0]
  R2_CS_decile_m[group]=CSR2_mom[group][1]

#############################################################
old_keys = list(R2_CS_m.keys())
new_keys = {'Toutes_vars': 'Ttes_vars','Tous_groupes':'ts_gp','Marche_de_travail':'Mr Tv', 'Logement':'Lgm', 'Conso_Ordres':'Cons Ord', 'Monnaie_credit':'Mn Cr', 'Taux_interet_change':'Int Ch', 'Prix':'Prix', 'Marches_Boursiers':'Mr Brs', 'Autres':'Autres',None:'Sans Mac', 'Production_Revenus':'Pr Rev'}

for old in old_keys:
   R2_CS_decile_m[new_keys[old]] = R2_CS_decile_m.pop(old)
   R2_CS_m[new_keys[old]] = R2_CS_m.pop(old)
########################



df_m = pd.DataFrame(R2_CS_decile_m)

df_m = df_m.round(decimals=3)
df_m.insert(0, 'decile', list(range(1, 11)))

new_row_m  = pd.DataFrame([R2_CS_m])
new_row_m.insert(0, 'decile', 'Tout')

df_m = pd.concat([df_m, new_row_m], ignore_index=True) # Ajoutez le DataFrame r2_cs au DataFrame r2_cs_decile en utilisant "Tour" comme index.



fig, ax = plt.subplots(figsize=(11, 6))

ax.axis('tight')
ax.axis('off')

# s'assurer que les entiers restent entiers
cellText = []
for row in df_m.itertuples(index=False):
    formatted_row = [f"{x:.3f}" if isinstance(x, float) else f"{x}" for x in row]
    cellText.append(formatted_row)

# Créer table
table = ax.table(cellText=cellText, colLabels=df_m.columns, cellLoc='center', loc='center')


plt.subplots_adjust(top=0.95)


#plt.suptitle('R2_CS pour les  portefeuilles de déciles triés par moment avec différents groupes macro', fontsize=12, y=0.85)

table.auto_set_font_size(False)
for key, cell in table.get_celld().items():
    if key[0] == 0:  
        cell.set_fontsize(9)
    else:
        cell.set_fontsize(9)

plt.savefig('final_results/figs/R2_CS,portefeuilles de deciles tries par momentum.jpg', bbox_inches='tight')

plt.show()






###################################### stat portefeuilles de déciles triés par size  ##########################"
############## Just pour abrévier les noms.

R2_CS = {}
R2_CS_decile={}

for group in CSR2_size.keys(): #  r2_cs et r2_cs_decile
  R2_CS[group] =CSR2_size[group][0]
  R2_CS_decile[group]=CSR2_size[group][1]

#############################################################"

#############################################################
old_keys = list(R2_CS.keys())
new_keys = {'Toutes_vars': 'Ttes_vars','Tous_groupes':'ts_gp','Marche_de_travail':'Mr Tv', 'Logement':'Lgm', 'Conso_Ordres':'Cons Ord', 'Monnaie_credit':'Mn Cr', 'Taux_interet_change':'Int Ch', 'Prix':'Prix', 'Marches_Boursiers':'Mr Brs', 'Autres':'Autres',None:'Sans Mac', 'Production_Revenus':'Pr Rev'}

for old in old_keys:
   R2_CS_decile[new_keys[old]] = R2_CS_decile.pop(old)
   R2_CS[new_keys[old]] = R2_CS.pop(old)
########################



df = pd.DataFrame(R2_CS_decile)

df = df.round(decimals=3)
df.insert(0, 'decile', list(range(1, 11)))

new_row  = pd.DataFrame([R2_CS])
new_row.insert(0, 'decile', 'Tout')

df = pd.concat([df, new_row], ignore_index=True) # Ajoutez le DataFrame r2_cs au DataFrame r2_cs_decile en utilisant "Tour" comme index.



fig, ax = plt.subplots(figsize=(11, 6))

ax.axis('tight')
ax.axis('off')

# s'assurer que les entiers restent entiers
cellText = []
for row in df.itertuples(index=False):
    formatted_row = [f"{x:.3f}" if isinstance(x, float) else f"{x}" for x in row]
    cellText.append(formatted_row)

# Créer table
table = ax.table(cellText=cellText, colLabels=df.columns, cellLoc='center', loc='center')


plt.subplots_adjust(top=0.95)


#plt.suptitle('R2_CS pour les  portefeuilles de déciles triés par momentum avec différents groupes macro', fontsize=12, y=0.85)

table.auto_set_font_size(False)
for key, cell in table.get_celld().items():
    if key[0] == 0:  
        cell.set_fontsize(9)
    else:
        cell.set_fontsize(9)

plt.savefig('final_results/figs/R2_CS,portefeuilles de deciles tries par size.jpg', bbox_inches='tight')

plt.show()