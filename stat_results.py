# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:32:53 2024

@author: Otman CH
"""

import pickle
from util import sharp, decile_portfolios,long_short_portfolio,DecileStatistics
from Load_data import Load_data
import pandas as pd
import matplotlib.pyplot as plt
from Plotting import plot_multi_dec_port_basedOn_pred_ret,plot_multi_L_S_portf_ret,plot_macro_probs






############################################proba Regime #####################################################################""    
      
plot_macro_probs([('macro_probabilities/macro_tr_prob_avec_retard.pkl', 'macro_probabilities/macro_test_prob_avec_retard.pkl')], save=True, lag= 'avec autoregression')
plot_macro_probs([('macro_probabilities/macro_tr_prob_sans_retard.pkl', 'macro_probabilities/macro_test_prob_sans_retard.pkl')], save=True, lag= 'sans autoregression')  







################# Obtenir les données de test et les prédictions, puis effectuez les calculs nécessaires. ###################################"
train_tmp_path = "datasets/char/Char_train.npz"
test_tmp_path  = "datasets/char/Char_test.npz"
valid_tmp_path = "datasets/char/Char_valid.npz" 

macro_paths = ('macro_probabilities/macro_tr_prob_sans_retard.pkl','macro_probabilities/macro_test_prob_sans_retard.pkl') 
  
groupes = ['Toutes_vars', 'Tous_groupes', 'Monnaie_credit','Marche_de_travail', 'Production_Revenus',  'Logement',
           'Conso_Ordres',  'Taux_interet_change', 'Prix', 'Marches_Boursiers', 'Autres', None]

   

decile_portf_on_ret = {}
ls_returns = {}
stat_sorted = {}
sharps = {}

sorting_var = 'r12_2'

for groupe in groupes:
      data = Load_data(train_tmp_path, test_tmp_path, valid_tmp_path,macro_paths, groupe)
      Xtest, Ytest, mask = data.get_test()
      char_names = data.get_var_names()
    
      with open(f'final_results/results/{groupe}_pred.pkl', 'rb') as f:
        if groupe == 'Tous_groupes' : pred = pickle.load(f)
        else :                      pred = pickle.load(f)
      sharps[groupe] = sharp(pred,Ytest,mask)
      decile_portf_on_ret[groupe] = decile_portfolios(pred, Ytest, mask, deciles=10)[0]
      ls_returns[groupe] = long_short_portfolio(pred, Ytest, mask, low=0.1, high=0.1, normalize=True)
    
      var2idx = {var:idx for idx, var in enumerate(char_names)}
      char_mask= Xtest[:,var2idx[sorting_var]]
      stat_sorted[groupe] = DecileStatistics(char_mask, Ytest, pred, mask, decile=10)
    
      del data, Xtest, mask, Ytest, pred
      
      
   


###################################### portf decile  basé sur pred de rend ############################################################

potf_ret_list = list(decile_portf_on_ret.values())
macro_group_list = list(decile_portf_on_ret.keys())

plot_multi_dec_port_basedOn_pred_ret(potf_ret_list, macro_group_list, save=True)






#########################################portf LS  basé sur pred de rend ###########################################################

ls_returns_list = list(ls_returns.values())
macro_group_list = list(ls_returns.keys())

plot_multi_L_S_portf_ret(ls_returns_list, macro_group_list, save=True)





############################### ratio de Sharpe  LS portf basé sur pred #############################""

if None in sharps:
    sharps['Sans macro'] = sharps.pop(None)

# Create a DataFrame from the dictionary
df = pd.DataFrame(list(sharps.items()), columns=['groupe macro', 'sharpe'])
df = df.sort_values(by='sharpe',ascending=False)
df = df.round(decimals=3)
fig, ax = plt.subplots(figsize=(10, 6))

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Adjust layout to make room for the title
plt.subplots_adjust(top=0.9)

# Add title to the table
#plt.suptitle('Ratio de sharpe OOS L_C portf', fontsize=12, y=0.85)

plt.savefig('final_results/figs/Table_ratios_de_sharpe.jpg', bbox_inches='tight')





###################################### stat portefeuilles de déciles triés par momentum  ##########################"
stat_sorted # diffrent stats 
R2_CS = {}
R2_CS_decile={}
for group in stat_sorted.keys(): #  r2_cs et r2_cs_decile
  R2_CS[group] =stat_sorted[group][2]
  R2_CS_decile[group]=stat_sorted[group][5]



############## Utilisez de nouvelles clés pour l'espace dans les figures.
new_keys = {'Toutes_vars': 'Ttes_vars','Tous_groupes':'ts_gp','Marche_de_travail':'Mr Tv', 'Logement':'Lgm', 'Conso_Ordres':'Cons Ord', 'Monnaie_credit':'Mn Cr', 'Taux_interet_change':'Int Ch', 'Prix':'Prix', 'Marches_Boursiers':'Mr Brs', 'Autres':'Autres',None:'Sans Mac', 'Production_Revenus':'Pr Rev'}

old_keys = list(R2_CS_decile.keys())

for old in old_keys:

   R2_CS_decile[new_keys[old]] = R2_CS_decile.pop(old)
   R2_CS[new_keys[old]] = R2_CS.pop(old)

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

plt.savefig('final_results/figs/R2_CS,portefeuilles de deciles tries par momentum.jpg', bbox_inches='tight')

plt.show()