# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:32:05 2024

@author: Otman CH
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


################## plot decile portfolio sorted based on predicted returns###############################################

def plot_dec_port_basedOn_pred_ret(potf_ret, macro_groupe, save = False):
    dates = pd.date_range(end='20161201', periods=potf_ret.shape[0], freq='MS')
    
    returns = pd.DataFrame(potf_ret, columns=[f'Decile {i}' for i in range(1, 11)], index=dates)
    
    
    plt.figure(figsize=(14, 8))
    
    for decile in returns.columns:
        s = returns[decile]
        s_cumsum = s.cumsum()
        plt.scatter(s_cumsum.index, s_cumsum / s.std(), s=10, label=decile)
    
    plt.title('Cumulative Returns of Decile Portfolios, ' + 'macro groupe:' + macro_groupe)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Excess Return based on R_hat')
    plt.legend(title='Decile Portfolios')
    plt.grid(True)
    if save: plt.savefig('plots/'+'Cumulative Returns of Decile Portfolios'+macro_groupe+'.pdf') 
    plt.show()
    

##################################long_short portfolio returns####################################################""

def plot_L_S_portf_ret(ls_returns, macro_groupe, save = False):
    
    dates = pd.date_range(end='20161201', periods=ls_returns.shape[0], freq='M')
    long_short_returns = pd.Series(ls_returns, index=dates, name='Long-Short')
    cumulative_returns = long_short_returns.cumsum()
    
    # Plotting the cumulative returns
    plt.figure(figsize=(14, 8))
    
    plt.scatter(cumulative_returns.index, cumulative_returns/cumulative_returns.std(), s = 10, label='Long-Short Portfolio')
    
    plt.title('Cumulative Returns of Long-Short Portfolio, ' + 'macro groupe:' + macro_groupe)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend(title='Portfolio')
    plt.grid(True)
    if save: plt.savefig('plots/'+'Cumulative Returns of Long-Short Portfolio'+macro_groupe+'.pdf') 
    plt.show()
    
    
    

###############################long_short  a part########################################"

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
    
    plt.title('Cumulative Returns of Long, Short Portfolios, ' + 'macro groupe:' + macro_groupe)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend(title='Portfolio')
    plt.grid(True)
    if save: plt.savefig('plots/'+'Cumulative Returns of Long and Short Portfolios'+macro_groupe+'.pdf') 
    plt.show()
    
    

############################ multiple returns #######################################""

def plot_multi_dec_port_basedOn_pred_ret(potf_ret_list, macro_group_list, save=False):
    n = len(potf_ret_list)
    if n != len(macro_group_list):
        raise ValueError("Length of potf_ret_list and macro_group_list must be the same.")

    # Calculate the number of rows and columns for the subplots
    ncols = 2
    nrows =  (n + 1) // ncols  # Ensures enough rows for all plots

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 8 * nrows))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, (potf_ret, macro_groupe) in enumerate(zip(potf_ret_list, macro_group_list)):
        dates = pd.date_range(end='20161201', periods=potf_ret.shape[0], freq='MS')
        returns = pd.DataFrame(potf_ret, columns=[f'Decile {j}' for j in range(1, 11)], index=dates)

        for decile in returns.columns:
            s = returns[decile]
            s_cumsum = s.cumsum()
            axs[i].scatter(s_cumsum.index, s_cumsum , s=10, label=decile)

        axs[i].set_title('Cumulative Returns of Decile Portfolios sorted based on pred, ' + ' \n macro group: ' +  str(macro_groupe), fontsize=6)
        axs[i].set_xlabel('Time', fontsize=6)
        axs[i].set_ylabel('Cumulative Return based on ret pred', fontsize=6)
        axs[i].legend(title='Decile Portfolios',title_fontsize=6, fontsize=4)
        axs[i].grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if save:
        fig.savefig('final_results/figs/Cumulative_Returns_of_diff_Decile_Portfolios_all.pdf')

    plt.show()


    
    
################################  plot multiple long short portf################################"




def plot_multi_L_S_portf_ret(ls_returns_list, macro_group_list, save=False):
  if len(ls_returns_list) != len(macro_group_list):
      raise ValueError("Length of ls_returns_list and macro_group_list must be the same.")
      
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
                 label=f'Long-Short Portfolio ({macro_group})')

  plt.title('Cumulative Returns of Long-Short Portfolios',fontsize=10)
  plt.xlabel('Time')
  plt.ylabel('Cumulative Return')
  plt.legend(title='Portfolio', title_fontsize=8, fontsize=6)
  plt.grid(True)

  if save:
      plt.savefig('final_results/figs//Cumulative_Returns_of_diff_Long_Short_Portfolios.pdf')

  plt.show()
    
    
    
    
    