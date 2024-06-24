# -*- coding: utf-8 -*-
"""
Created on Fri May 10 02:32:04 2024

@author: Otman CH


Ces fonctions sont destinées à calculer les différentes statistics en se basant sur les prédictions.
La construction des portfeuilles suit la même technique utlisée par Chen et al. (2023)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Porfeuille lon court basé sur la prédiction du rendement 
def long_short_portfolio(pred, R, mask, low=0.1, high=0.1, normalize=True):
    
    
    R_mask = R[mask]
    N_i = np.sum(mask.astype(int), axis=1) # nombre d'actifs par mois
    N_i_cumsum = np.cumsum(N_i) # nombre cumulatif d'actifs par mois:  indices pour repérer ou on va appliquer split
    w_split = np.split(pred, N_i_cumsum)[:-1] # liste : predictions présents  pour chaque mois
    R_split = np.split(R_mask, N_i_cumsum)[:-1] # observations présentes pour chaque mois

    portfolio_returns = []

    for j in range(len(N_i)): # itérer sur les mois
        R_j = R_split[j] #  actif present au j eme mois
        w_j = w_split[j] # pred au j eme mois

        R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])] # couples d'observations et de prédictions (par actif) du même mois
        R_w_j_sorted = sorted(R_w_j, key=lambda t:t[1]) # trier (obs, pred) basé sur pred w_j = R_w_j[1]
        n_low = int(low * N_i[j])
        n_high = int(high * N_i[j])

        if n_high == 0.0:
            portfolio_return_high = 0.0
        else:
            portfolio_return_high = 0.0
            value_sum_high = 0.0
            for k in range(n_high):
                portfolio_return_high += R_w_j_sorted[-k-1][0] * R_w_j_sorted[-k-1][2]
                value_sum_high += R_w_j_sorted[-k-1][2]
            if normalize:
                portfolio_return_high /= value_sum_high

        if n_low == 0:
            portfolio_return_low = 0.0
        else:
            portfolio_return_low = 0.0
            value_sum_low = 0.0
            for k in range(n_low):
                portfolio_return_low += R_w_j_sorted[k][0] * R_w_j_sorted[k][2]
                value_sum_low += R_w_j_sorted[k][2]
            if normalize:
                portfolio_return_low /= value_sum_low

        portfolio_returns.append(portfolio_return_high - portfolio_return_low)
    return np.array(portfolio_returns)
################################################################################



####################################################################################


def sharp(R_pred, R, mask):
    
        portfolio = long_short_portfolio(R_pred,R, mask) # equi-pondérés : equally weighted
        sharp = np.mean((portfolio)/ portfolio.std())
        return sharp
    
    
###########################################################################################"
      
       
# Porfeuilles de décile triés et basés sur la prédiction du rendement 
def decile_portfolios(w, returns, mask, deciles=10):
    #calculer le rendement du portfeuille de décile pour le graph 
    returns = returns[mask]
    num_assets = np.sum(mask.astype(int), axis=1)
    cumulative_assets = np.cumsum(num_assets)
    pred_splits = np.split(w, cumulative_assets)[:-1]
    return_splits = np.split(returns, cumulative_assets)[:-1]
    

    portfolio_returns = []
    portfolio_preds = []
    for month in range(mask.shape[0]):
        asset_returns = return_splits[month] 
        asset_pred = pred_splits[month]


        asset_info = [(asset_returns[i], asset_pred[i], 1) for i in range(num_assets[month])]

        asset_info_sorted = sorted(asset_info, key=lambda x: x[1])

        assets_per_decile = num_assets[month] // deciles

        decile_returns = []
        decile_preds = []
        for i in range(deciles):
            decile_return = 0.0
            value_sum = 0.0
            decile_pred=0.0

            for j in range(assets_per_decile):
                index = i * assets_per_decile + j
                decile_return += asset_info_sorted[index][0] * asset_info_sorted[index][2]
                decile_pred += asset_info_sorted[index][1] * asset_info_sorted[index][2]
                value_sum += asset_info_sorted[index][2]
                

            decile_returns.append(decile_return / value_sum)
            decile_preds.append(decile_pred / value_sum)

        portfolio_returns.append(decile_returns)
        portfolio_preds.append(decile_preds)

    return (np.array(portfolio_returns),np.array(portfolio_preds))

#######################################################################################"


def R2(R, residual, cross_sectional=False):
	if cross_sectional:
		return 1 - np.mean(np.square(residual.mean(axis=0))) / np.mean(np.square(R.mean(axis=0)))
	else:
		return 1 - np.mean(np.square(residual)) / np.mean(np.square(R))
    
#############################################################################################"    
  


# Calculer les statistics de portfeuilles de déciles trié on se basant sur une charactéristique: char_masked
def DecileStatistics(char_masked, R, pred, mask, decile=10):
    
    R = R[mask]

    # Calculer splits , R_split, pred, et char_masked 
    splits = np.cumsum(np.sum(mask, axis=1))[:-1]
    R_split, pred_split, char_split = np.split(R, splits), np.split(pred, splits), np.split(char_masked, splits)

    # resultats de décile 
    R_decile = np.zeros((mask.shape[0], decile))
    pred_decile = np.zeros((mask.shape[0], decile))

    for month in range(mask.shape[0]):
        # trier en se basant sur  char_t
        sorted_data = sorted(zip(R_split[month], pred_split[month], char_split[month]), key=lambda x: x[2])
        decile_splits = np.array_split(sorted_data, decile)

        # Moyenne pour chaque décile
        for i, group in enumerate(decile_splits):
            R_decile[month, i] = np.mean([x[0] for x in group])
            pred_decile[month, i] = np.mean([x[1] for x in group])

   
    R2_CS,R2_CS_decile = CS_R2_decile(R_decile, pred_decile)

    return R2_CS, R2_CS_decile



def CS_R2_decile(R_decile, pred_decile):
    decile = R_decile.shape[1]
    residual_decile = R_decile - pred_decile
    R2_CS_decile = np.array([R2(R_decile[:, i], residual_decile[:, i], cross_sectional=True) for i in range(decile)])
    R2_CS = R2(R_decile, residual_decile, cross_sectional=True)
    
    return R2_CS,R2_CS_decile

    
    