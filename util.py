# -*- coding: utf-8 -*-
"""
Created on Fri May 10 02:32:04 2024

@author: Otman CH


Function for matrics computations on the FFN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def long_short_portfolio(pred, R, mask, low=0.1, high=0.1, normalize=True):
    
    
    R_mask = R[mask]
    N_i = np.sum(mask.astype(int), axis=1) # number of assets per month
    N_i_cumsum = np.cumsum(N_i) # cumulative number of assets per month: sections to indicate where the pred will be split
    w_split = np.split(pred, N_i_cumsum)[:-1] # list containing present pred per months
    R_split = np.split(R_mask, N_i_cumsum)[:-1] # list of present obs per month

    portfolio_returns = []

    for j in range(len(N_i)): # iterate over months
        R_j = R_split[j] #  assets present in the j month
        w_j = w_split[j] # pred of the j month

        R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])] # couples of observable and predictions (assets) of the same month
        R_w_j_sorted = sorted(R_w_j, key=lambda t:t[1]) # sorted (obs, pred) based on pred w_j = R_w_j[1]
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
    

        portfolio = long_short_portfolio(R_pred,R, mask) # equally weighted
        sharp = np.mean((portfolio)/ portfolio.std())
        return sharp
###########################################################################################"
      
       

def decile_portfolios(w, returns, mask, deciles=10):
    #compute returns of decile porfolios to plot 
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
  



    
    
def calculateStatistics(w, I, R, mask):
	R_hat, residual, mask, R = decomposeReturn(w,  R, mask)
                                                                                                                                                            
	T_i = np.sum(mask, axis=0)
	N_t = np.sum(mask, axis=1)
	stat1 = 1 - np.mean(np.square(residual).sum(axis=1) / N_t) / np.mean(np.square(R * mask).sum(axis=1) / N_t) # EV
	stat2 = 1 - np.mean(np.square(residual.sum(axis=0) / T_i)) / np.mean(np.square((R * mask).sum(axis=0) / T_i)) # XS-R2
	stat3 = 1 - np.mean(np.square(residual.sum(axis=0) / T_i) * T_i) / np.mean(np.square((R * mask).sum(axis=0) / T_i) * T_i) # XS-R2 weighted
	return stat1, stat2, stat3
##############################################################################################""ù



def decomposeReturn(w, R, mask):

    R_reshape = R[mask]
    #sum numper of assets per month
    splits = np.sum(mask, axis=1).cumsum()[:-1]
    w_list = np.split(w, splits)
    R_list = np.split(R_reshape, splits)
    R_hat_list = []
    residual_list = []
  
    for R_i, w_i in zip(R_list, w_list):
  			
  			residual_i = R_i - w_i
  			R_hat_list.append(w_i)
  			residual_list.append(residual_i)
      
  
    R_hat = np.zeros_like(mask, dtype=float)
    residual = np.zeros_like(mask, dtype=float)
    R_hat[mask] = np.concatenate(R_hat_list)
    residual[mask] = np.concatenate(residual_list)
    return R_hat, residual, mask, R

###################################################################################################"
    
    
def UnexplainedVariation(R, residual):
   return np.mean(np.square(residual)) / np.mean(np.square(R))   


def FamaMcBethAlpha(residual):
	return np.sqrt(np.mean(np.square(residual.mean(axis=0))))


def DecileStatistics(char_masked, R, pred, mask, decile=10):
    # Filter R using the mask
    R = R[mask]

    # Compute splits and split R, pred, and char_masked accordingly
    splits = np.cumsum(np.sum(mask, axis=1))[:-1]
    R_split, pred_split, char_split = np.split(R, splits), np.split(pred, splits), np.split(char_masked, splits)

    # Initialize arrays for decile results
    R_decile = np.zeros((mask.shape[0], decile))
    pred_decile = np.zeros((mask.shape[0], decile))

    for month in range(mask.shape[0]):
        # Sort and split data into deciles based on char_t
        sorted_data = sorted(zip(R_split[month], pred_split[month], char_split[month]), key=lambda x: x[2])
        decile_splits = np.array_split(sorted_data, decile)

        # Compute means for each decile
        for i, group in enumerate(decile_splits):
            R_decile[month, i] = np.mean([x[0] for x in group])
            pred_decile[month, i] = np.mean([x[1] for x in group])

    # Calculate residuals
    residual_decile = R_decile - pred_decile

    # Compute statistics for each decile
    UV_decile = np.array([UnexplainedVariation(R_decile[:, i], residual_decile[:, i]) for i in range(decile)])
    Alpha_decile = np.array([FamaMcBethAlpha(residual_decile[:, i]) for i in range(decile)])
    R2_CS_decile = np.array([R2(R_decile[:, i], residual_decile[:, i], cross_sectional=True) for i in range(decile)])

    # Compute overall statistics
    UV = UnexplainedVariation(R_decile, residual_decile)
    Alpha = FamaMcBethAlpha(residual_decile)
    R2_CS = R2(R_decile, residual_decile, cross_sectional=True)

    return UV, Alpha, R2_CS, UV_decile, Alpha_decile, R2_CS_decile



    
    