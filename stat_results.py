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





################# get test data and predictions and do calculations ###################################"
train_tmp_path = "datasets/char/Char_train.npz"
test_tmp_path  = "datasets/char/Char_test.npz"
valid_tmp_path = "datasets/char/Char_valid.npz" 

macro_paths = ('macro_probabilities/macro_tr_prob.pkl','macro_probabilities/macro_test_prob.pkl')   
groupes = ['all_vars', 'all_groups', 'money_credit','Labor_market', 'Output_Income',  'Housing',
           'Consumption_OR',  'interest_exchange_rates', 'Prices', 'Stock_markets', 'Other', None]

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
        if groupe == 'all_groups' : pred = pickle.load(f)
        else :                      pred = pickle.load(f)
      sharps[groupe] = sharp(pred,Ytest,mask)
      decile_portf_on_ret[groupe] = decile_portfolios(pred, Ytest, mask, deciles=10)[0]
      ls_returns[groupe] = long_short_portfolio(pred, Ytest, mask, low=0.1, high=0.1, normalize=True)
    
      var2idx = {var:idx for idx, var in enumerate(char_names)}
      char_mask= Xtest[:,var2idx[sorting_var]]
      stat_sorted[groupe] = DecileStatistics(char_mask, Ytest, pred, mask, decile=10)
    
      del data, Xtest, mask, Ytest, pred
      
      
############################################ Regime proba#####################################################################""    
      
plot_macro_probs([('macro_probabilities/macro_tr_prob.pkl', 'macro_probabilities/macro_test_prob.pkl')], save=True, lag= 'with_lag')
plot_macro_probs([('macro_probabilities/macro_tr_prob_no_lag.pkl', 'macro_probabilities/macro_test_prob_no_lag.pkl')], save=True, lag= 'no_lag')     
###################################### decile portf based on return ############################################################

potf_ret_list = list(decile_portf_on_ret.values())
macro_group_list = list(decile_portf_on_ret.keys())

plot_multi_dec_port_basedOn_pred_ret(potf_ret_list, macro_group_list, save=True)


######################################### LS portf based on ret ###########################################################

ls_returns_list = list(ls_returns.values())
macro_group_list = list(ls_returns.keys())

plot_multi_L_S_portf_ret(ls_returns_list, macro_group_list, save=True)


############################### Sharp ratio LS portf based on ret #############################""

if None in sharps:
    sharps['without macro'] = sharps.pop(None)

# Create a DataFrame from the dictionary
df = pd.DataFrame(list(sharps.items()), columns=['macro groupe', 'sharp'])
df = df.round(decimals=3)
fig, ax = plt.subplots()

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Adjust layout to make room for the title
plt.subplots_adjust(top=0.9)

# Add title to the table
plt.suptitle('Sharp Ratio OOS L_S portf', fontsize=12, y=0.85)

plt.savefig('final_results/figs/sharp_ratios_table.jpg', bbox_inches='tight')


###################################### stat of decile sorted portf based on momentum ##########################"
stat_sorted # diffrent stats 
R2_CS = {}
R2_CS_decile={}
for group in stat_sorted.keys(): # get r2_cs and r2_cs_decile
  R2_CS[group] =stat_sorted[group][2]
  R2_CS_decile[group]=stat_sorted[group][5]



############## use new keys for space in figs 
new_keys = {'all_vars': 'all_vars','all_groups':'all_gr','Labor_market':'Lb Mrk', 'Housing':'Hous', 'Consumption_OR':'Cons Qr', 'money_credit':'Mny Cr', 'interest_exchange_rates':'Int Ex', 'Prices':'Pce', 'Stock_markets':'Stk Mrt', 'Other':'Other',None:'no Mac', 'Output_Income':'Out In'}

old_keys = list(R2_CS_decile.keys())

for old in old_keys:

   R2_CS_decile[new_keys[old]] = R2_CS_decile.pop(old)
   R2_CS[new_keys[old]] = R2_CS.pop(old)

df = pd.DataFrame(R2_CS_decile)

df = df.round(decimals=3)
df.insert(0, 'decile', list(range(1, 11)))

new_row  = pd.DataFrame([R2_CS])
new_row.insert(0, 'decile', 'all')

df = pd.concat([df, new_row], ignore_index=True) # add r2_cs dataframe to r2_cs_decile dataframe using all for the index



# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Hide axes
ax.axis('tight')
ax.axis('off')

# Format the cellText to ensure integers remain integers
cellText = []
for row in df.itertuples(index=False):
    formatted_row = [f"{x:.3f}" if isinstance(x, float) else f"{x}" for x in row]
    cellText.append(formatted_row)

# Create the table
table = ax.table(cellText=cellText, colLabels=df.columns, cellLoc='center', loc='center')

# Adjust layout to make room for the title
plt.subplots_adjust(top=0.85)

# Add title to the table
plt.suptitle('CS_R2 deciles of momentum sorted porfolios with diff groups of macro', fontsize=12, y=0.85)

table.auto_set_font_size(False)
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Header row
        cell.set_fontsize(10)
    else:
        cell.set_fontsize(9)

plt.savefig('final_results/figs/CS_R2 deciles of momentum sorted porfolios with diff groups of macro.jpg', bbox_inches='tight')
# Show the plot
plt.show()