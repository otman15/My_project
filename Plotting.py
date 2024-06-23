# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:32:05 2024

@author: Otman CH
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas_datareader.data as web


'''
Ces fonctions aident à produir les différents graphes et les tableaux
'''

#################################################Etats des régimes#################################################"
def plot_macro_probs(paths, save=False, lag ='sans auto_regression'):
    data = web.DataReader('USREC', 'fred', start='1967-01-01', end='2016-12-31')
    data = data.reset_index()
    recession_starts = data['DATE'][data['USREC'].diff() == 1]
    recession_ends = data['DATE'][data['USREC'].diff() == -1]
    
    
    # Charger les dictionnaires à partir des fichiers pickle
    time_series_tr = {}
    time_series_test = {}
    for path_tr, path_test in paths:
        with open(path_tr, 'rb') as f_tr, open(path_test, 'rb') as f_test:
            time_series_tr.update(pickle.load(f_tr))
            time_series_test.update(pickle.load(f_test))

    num_series = len(time_series_tr)

    num_months = len(time_series_tr['Production_Revenus']) + len(time_series_tr['Production_Revenus'])
    dates = pd.date_range(end='20161201', periods=num_months, freq='MS')
    
    
    num_rows = (num_series + 1) // 2

   
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, num_rows * 3.5))

    
    axs = axs.flatten()

    # Itérer à travers le dictionnaire et tracer chaque série temporelle
    for i, key in enumerate(time_series_tr.keys()):
        tr = time_series_tr[key]
        test = time_series_test[key]
        val = np.concatenate((tr, test))
        axs[i].scatter(dates, val)
        axs[i].set_title(key + ' ' + lag , fontsize=16)
        for start, end in zip(recession_starts, recession_ends):
            axs[i].axvspan(start, end, color='grey', alpha=0.5)

    for i in range(num_series, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()

    if save:
        plt.savefig('final_results/figs/macro_prob '+ lag +'.png')
    
    plt.show()


################## Tracer le portefeuille décile trié en fonction des rendements prédits.###############################################

def plot_dec_port_basedOn_pred_ret(potf_ret, macro_groupe, save = False):
    dates = pd.date_range(end='20161201', periods=potf_ret.shape[0], freq='MS')
    
    returns = pd.DataFrame(potf_ret, columns=[f'Decile {i}' for i in range(1, 11)], index=dates)
    
    
    plt.figure(figsize=(14, 8))
    
    for decile in returns.columns:
        s = returns[decile]
        s_cumsum = s.cumsum()
        plt.scatter(s_cumsum.index, s_cumsum / s.std(), s=10, label=decile)
    
    plt.title('Rendements cumulatifs des portefeuilles déciles, ' + 'macro groupe:' + macro_groupe)
    plt.xlabel('Time')
    plt.ylabel('Rendement cumulatif basé sur \( \hat{R} \)t')
    plt.legend(title='Portefeuilles déciles')
    plt.grid(True)
    if save: plt.savefig('plots/'+'Rendements cumulatifs des portefeuilles déciles'+macro_groupe+'.pdf') 
    plt.show()
    

#################################Rendements du portefeuille long-short####################################################""

def plot_L_S_portf_ret(ls_returns, macro_groupe, save = False):
    
    dates = pd.date_range(end='20161201', periods=ls_returns.shape[0], freq='M')
    long_short_returns = pd.Series(ls_returns, index=dates, name='Long-Short')
    cumulative_returns = long_short_returns.cumsum()
    
    # Plotting the cumulative returns
    plt.figure(figsize=(14, 8))
    
    plt.scatter(cumulative_returns.index, cumulative_returns/cumulative_returns.std(), s = 10, label='Long-Short Portfolio')
    
    #plt.title('Rendements cumulatifs du portefeuille long-court, ' + 'macro groupe:' + macro_groupe)
    plt.xlabel('Temps')
    plt.ylabel('Rendement Cumulatif')
    plt.legend(title='portefeuille LC')
    plt.grid(True)
    if save: plt.savefig('plots/'+'Rendements cumulatifs du portefeuille long-short'+macro_groupe+'.jpg') 
    plt.show()
    
    
    

###############################long_short à part########################################"

def long_and_short(long_returns, short_returns,long_short_returns,macro_groupe, save = False):

    
    dates = pd.date_range(end='20161201', periods=long_returns.shape[0], freq='M')
    
    cumulative_long_returns=pd.Series(long_returns.cumsum(), index=dates, name='Long')
    cumulative_short_returns=pd.Series(short_returns.cumsum(), index=dates, name='Short')
    #cumulative_long_short_returns=pd.Series(long_short_returns.cumsum(), index=dates, name='Long_short')
    
    
    # Plotting the cumulative returns
    plt.figure(figsize=(14, 8))
    
    plt.plot(cumulative_long_returns.index, cumulative_long_returns/long_returns.std(), label='Long Portfolio')
    plt.plot(cumulative_short_returns.index, cumulative_short_returns/short_returns.std(), label='Short Portfolio')
    #plt.plot(cumulative_long_short_returns.index, cumulative_long_short_returns, label='Long-Short Portfolio')
    
    plt.title('Rendements cumulatifs des portefeuilles long et court, ' + 'macro groupe:' + macro_groupe)
    plt.xlabel('Temps')
    plt.ylabel('Rendement Cumulatif')
    plt.legend(title='portefeuilles')
    plt.grid(True)
    if save: plt.savefig('plots/'+'Rendements cumulatifs des portefeuilles long et court'+macro_groupe+'.jpg') 
    plt.show()
    
    

############################ Rendements multiples #######################################""

def plot_multi_dec_port_basedOn_pred_ret(potf_ret_list, macro_group_list, save=False):
    n = len(potf_ret_list)
    if n != len(macro_group_list):
        raise ValueError("La longueur de potf_ret_list et macro_group_list doit être la même..")

    # Calculate the number of rows and columns for the subplots
    ncols = 3
    nrows =  (n + 1) // ncols  # Ensures enough rows for all plots

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 7 * nrows))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, (potf_ret, macro_groupe) in enumerate(zip(potf_ret_list, macro_group_list)):
        dates = pd.date_range(end='20161201', periods=potf_ret.shape[0], freq='MS')
        returns = pd.DataFrame(potf_ret, columns=[f'Decile {j}' for j in range(1, 11)], index=dates)

        for decile in returns.columns:
            s = returns[decile]
            s_cumsum = s.cumsum()
            axs[i].scatter(s_cumsum.index, s_cumsum , s=10, label=decile)

        axs[i].set_title('groupe macro: ' +  str(macro_groupe), fontsize=15)
        axs[i].set_xlabel('Temps', fontsize=8)
        axs[i].set_ylabel('Rendement cumulatif basé sur les prédictions de rendement', fontsize=8)
        axs[i].legend(title='Portefeuilles déciles',title_fontsize=10, fontsize=8)
        axs[i].grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if save:
        fig.savefig('final_results/figs/Rendements cumulatifs des portefeuilles déciles triés en fonction des prédictions de rendement.jpg')

    plt.show()


    
    
################################  Tracer plusieurs portefeuilles long-short ################################"




def plot_multi_L_S_portf_ret(ls_returns_list, macro_group_list, save=False):
  if len(ls_returns_list) != len(macro_group_list):
      raise ValueError("La longueur de ls_returns_list et macro_group_list doit être la même.")
      
  colors = plt.cm.viridis(np.linspace(0, 1, len(ls_returns_list)))
  markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X']

  plt.figure(figsize=(10, 6))

  for i, (ls_returns, macro_group) in enumerate(zip(ls_returns_list, macro_group_list)):
        dates = pd.date_range(end='20161201', periods=ls_returns.shape[0], freq='M')
        long_short_returns = pd.Series(ls_returns, index=dates, name='Long-Short')
        cumulative_returns = long_short_returns.cumsum()

        # Plotting the cumulative returns with different colors and markers
        plt.plot(cumulative_returns.index, cumulative_returns, 
                 color=colors[i % len(colors)], 
                 marker=markers[i % len(markers)], 
                 linestyle='None', 
                 markersize=2, 
                 label=f'{macro_group}')

  #plt.title('Rendements cumulatifs des portefeuilles long-courts',fontsize=10)
  plt.xlabel('Temps')
  plt.ylabel('Rendement Cumulatif')
  plt.legend(title='groupe of macro', title_fontsize=10, fontsize=9)
  plt.grid(True)

  if save:
      plt.savefig('final_results/figs//Rendements cumulatifs des différents portefeuilles long-courts.png')

  plt.show()
    
    
    
    
    