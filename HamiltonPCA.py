
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


'''
Hamilton class:
    takes univariate or multivariate time seris, apply a hamilton filter and kim smoother
    and compute the filtered and smoothed probablity regimes.
    if data is used to compute filter params (training data), the kim smoother is used since
    it s supposed we have access to all data.
    
    if a new test data is given to predict new regime probabilites, we suppose we don't have 
    access to all the time series but only to present observations, so the kim smoother is not
    applied but the trained params and last probablity are used to infer new probablities.
'''


class HamiltonKimModel:
    def __init__(self, data, lag=True):

        if data.ndim == 1:  # Univariate time series
            data = data[np.newaxis, :]  # Convert to 2D array with one row
        self.lag = lag
        self.y = data[:, 1:]  # Include all elements except the first one for each series, to have a lagged element
        if self.lag:
            self.x = data[:, :-1]  # Include all elements except the last one for each series for lagged data 
        self.param = None


    
    def hamilton_filter(self, param=None, new_obs=None):
        # if param is not provided, this mean we are not optimizing but just predicting new prob on new data (test set)
        if param is None:
              param = self.param
        
        # if new observations are provided use it as the time series, and use the params to get initial first states probs.
        if new_obs is not None:
            
            n_series = self.y.shape[0]
            initial_probs = self.sp[-1]
            new_y = new_obs if new_obs.ndim > 1 else new_obs[np.newaxis, :]
            last_y = self.y[:, -1].reshape(n_series, 1)
            combined_y = np.hstack((last_y, new_y))
            if self.lag:
                self.x = combined_y[:, :-1]  # All but the last element
            self.y = combined_y[:, 1:]  # All but the first element
            E0_t, E1_t = initial_probs

        n_series = self.y.shape[0]
        
        # retrieve the params: const, mu and sigma
        if self.lag:
            const = param[:n_series*2].reshape(n_series, 2)
            beta = param[n_series*2:n_series*4].reshape(n_series, 2)
            sigma = param[n_series*4:n_series*6].reshape(n_series, 2)
            p7, p8 = param[-2:]
        else:
            const = param[:n_series*2].reshape(n_series, 2)
            sigma = param[n_series*2:n_series*4].reshape(n_series, 2)
            p7, p8 = param[-2:]

        nobs = self.y.shape[1]

        # Transition probabilities
        p00 = 1 / (1 + np.exp(-p7))
        p11 = 1 / (1 + np.exp(-p8))
        p01 = 1 - p00
        p10 = 1 - p11
        
        # if computing params (training) initiate the  first states probs
        if new_obs is None:
            
            E1_t = (1 - p00) / (2 - p11 - p00)
            E0_t = 1 - E1_t

        # Predicted state, filtered probability
        pred_st = np.zeros((nobs, 2))
        fp = np.zeros((nobs, 2))

        loglike = 0
        for t in range(nobs):
            # Previous state
            E0_t_1 = E0_t
            E1_t_1 = E1_t

            # Regression estimation
            mu0 = np.zeros(n_series)
            mu1 = np.zeros(n_series)
            if self.lag:
                for n in range(n_series):
                    mu0[n] = const[n, 0] + beta[n, 0] * self.x[n, t]
                    mu1[n] = const[n, 1] + beta[n, 1] * self.x[n, t]
            else:
                mu0 = const[:, 0]
                mu1 = const[:, 1]

            # Densities under the two regimes at t
            fy_g_st0 = 1
            fy_g_st1 = 1
            for n in range(n_series):# independent observations f(y1,y2,..) = f(y1)*f(y2)*....
                fy_g_st0 *= (1 / np.sqrt(2 * np.pi * sigma[n, 0] ** 2)) * np.exp(-((self.y[n, t] - mu0[n]) ** 2) / (2 * sigma[n, 0] ** 2))
                fy_g_st1 *= (1 / np.sqrt(2 * np.pi * sigma[n, 1] ** 2)) * np.exp(-((self.y[n, t] - mu1[n]) ** 2) / (2 * sigma[n, 1] ** 2))

            # Density of yt
            f_yt = E0_t_1 * p00 * fy_g_st0 + E0_t_1 * p01 * fy_g_st1 + E1_t_1 * p10 * fy_g_st0 + E1_t_1 * p11 * fy_g_st1

            if f_yt < 0 or np.isnan(f_yt):
                loglike = -100000000
                break

            # State at t+1 given t
            E0_t = (E0_t_1 * p00 * fy_g_st0 + E1_t_1 * p10 * fy_g_st0) / f_yt
            E1_t = (E0_t_1 * p01 * fy_g_st1 + E1_t_1 * p11 * fy_g_st1) / f_yt

            pred_st[t, :] = [E0_t_1, E1_t_1]  # Predicted states
            fp[t, :] = [E0_t, E1_t]  # Filtered probabilities

            loglike += np.log(f_yt)

        return {'loglike': -loglike, 'fp': fp, 'pred_st': pred_st}
    
    

    def kim_smoother(self):
        p7, p8 = self.param[-2:]

        nobs = self.y.shape[1]

        p00 = 1 / (1 + np.exp(-p7))
        p11 = 1 / (1 + np.exp(-p8))
        p01 = 1 - p00
        p10 = 1 - p11

        fp = np.zeros((nobs, 2))
        sp = np.zeros((nobs, 2))

        # get filtered estimates
        hf = self.hf
        # Get filtered outputs
        fp = hf['fp']  # Filtered probability

        T = nobs
        sp[T - 1, :] = fp[T - 1, :]

        # Iteration from T-1 to 1
        for is_ in range(T - 2, -1, -1):
            p1 = (sp[is_ + 1, 0] * fp[is_, 0] * p00) / (fp[is_, 0] * p00 + fp[is_, 1] * p10)
            p2 = (sp[is_ + 1, 1] * fp[is_, 0] * p01) / (fp[is_, 0] * p01 + fp[is_, 1] * p11)
            p3 = (sp[is_ + 1, 0] * fp[is_, 1] * p10) / (fp[is_, 0] * p00 + fp[is_, 1] * p10)
            p4 = (sp[is_ + 1, 1] * fp[is_, 1] * p11) / (fp[is_, 0] * p01 + fp[is_, 1] * p11)

            sp[is_, 0] = p1 + p2
            sp[is_, 1] = p3 + p4

        return sp

    def fit(self):
        if self.lag:
            n_series = self.y.shape[0]
            init_guess = np.ones(n_series * 6 + 2)
            init_guess[-2:] = [3.0, 2.9]
            bounds = [(None, None)] * (n_series * 6) + [(1e-6, None), (1e-6, None)]
        else:
            n_series = self.y.shape[0]
            init_guess = np.ones(n_series * 4 + 2)
            init_guess[-2:] = [3.0, 2.9]
            bounds = [(None, None)] * (n_series * 4) + [(1e-6, None), (1e-6, None)]


        mle = minimize(self.objective, init_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50000})
        return mle.x

    def objective(self, param):
        result = self.hamilton_filter(param=param)
        return result['loglike']

    def run(self):
        self.param = self.fit()
        self.hf = self.hamilton_filter()
        self.sp = self.kim_smoother()

    def get_sp_hf(self):# kim smooth prob and hamilton filter results: filtered prob, predicted states
        return {'smoothed_prob':self.sp, 'filtered_prob': self.hf['fp'], 'predicted_states':self.hf['pred_st']}


    def plot_results(self, dates= None):
        plt.figure(figsize=(8, 6))

        if all(dates) == None:
            plt.plot(self.sp[:, 0], label='Regime 1', color='green', linewidth=3, linestyle='-')
            plt.plot(self.sp[:, 1], label='Regime 2', color='blue', linewidth=3, linestyle='-')
        else:
            plt.plot(dates,self.sp[:, 0], label='Regime 1', color='green', linewidth=3, linestyle='-')
            plt.plot(dates,self.sp[:, 1], label='Regime 2', color='blue', linewidth=3, linestyle='-')
            n = 35 # Display every 10th label
            plt.xticks(dates[::n], rotation=45, fontsize=8)

        plt.title('Smoothed probability')
        plt.ylabel('Probability')
        plt.legend(loc='lower right')


        plt.tight_layout()
        plt.show()


################################################################################################################################################

################################################################################################################################################



'''
Group_macro_by_Class:
    this class is used to group macro data by groupe: 10 groupes including a groupe of all data(no grouping).
    it perform Pca on each group and yield the first component of each Pca groupe of macro
    Outpout a dictionnary of (groupe: PC1)
    it can also retrieve data and plot PC1
'''


class Group_macro_Pca:

    def __init__(self, data1):

        self.data1 = data1
        self.scaled = 'non scaled'
        self.data_dict = self.group()
       # self.PC1 = {}
       # self.explained_variance_ratio = {}
        self.groups = ['Output_Income', 'Labor_market', 'Housing','Consumption_OR' , 'money_credit',
                       'interest_exchange_rates', 'Prices', 'Stock_markets', 'Other', 'all_vars']

    def group(self):
        #group1 = data1.columns[1:20]
        Output_Income1 = ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS',  'IPFINAL','IPCONGD', 'IPDCONGD', 'IPNCONGD', \
                        'IPBUSEQ',  'IPMAT','IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S',  'IPFUELS', 'CUMFNS']

        #group2_names = data1.columns[20:48]
        Labor_market2 = ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN',
               'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx',
               'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP',
               'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE',
               'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008','CES2000000008','CES3000000008']


        #group3_names = data1.columns[48:58] # 49 ....  61

        Housing3 = ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE',
               'PERMITMW', 'PERMITS', 'PERMITW']

        #group4 = data1.columns[3:6]

        Consumption_OR4 = ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'AMDMNOx', 'AMDMUOx', 'BUSINVx', 'ISRATIOx']

        #group5 = data1.columns[62:72]

        money_credit5 = ['M1SL', 'M2SL', 'M2REAL', 'AMBSL', 'TOTRESNS', 'NONBORRES', 'BUSLOANS',
               'REALLN', 'NONREVSL', 'CONSPI', 'MZMSL', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST']

        #groupr6  = data1.columns[76:97]

        interest_exchange_r6 = ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA',
               'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM',
               'T10YFFM', 'AAAFFM', 'BAAFFM', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx',
               'EXCAUSx']

        #groupr7  = data1.columns[97:117]

        Prices7 = ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM',
               'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC',
               'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2',
               'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA',
               'DSERRG3M086SBEA']

        #group8 = data1.columns[72:76]

        Stock_markets8 = ['S&P 500', 'S&P: indust', 'S&P div yield', 'S&P PE ratio','VXOCLSx']



        self.data_dict = {
            'Output_Income': self.data1[Output_Income1], #'Output_Income': self.data1[Output_Income1].values
            'Labor_market': self.data1[Labor_market2],
            'Housing': self.data1[Housing3],
            'Consumption_OR': self.data1[Consumption_OR4],
            'money_credit': self.data1[money_credit5],
            'interest_exchange_rates': self.data1[interest_exchange_r6],
            'Prices': self.data1[Prices7],
            'Stock_markets': self.data1[Stock_markets8],
            'Other': self.data1.iloc[:,125:],
            'all_vars': self.data1.iloc[:,1:]

        }


    def get_data(self, variable_name=None):
       # use try catch here if variable_name == None : variable_name = self.groups
        self.group()
        if variable_name:
            if variable_name in self.data_dict:
                return self.data_dict[variable_name]
            else:
                print(f"Variable '{variable_name}' not found in the dataset.")
                return None
        else:
            return self.data_dict

    def get_Pca(self, scaled = 'non_scaled' ,groups=None):
        if scaled != 'scaled' and scaled != 'non_scaled':
            print('please enter scaled or non_scaled')
            return [],[]
        self.PC1 = {}
        self.explained_variance_ratio = {}
        # use try catch here , groups is a list
        if groups == None : groups = self.groups
        self.group()

        for group in groups:

            scaler = StandardScaler()
            data = self.data_dict[group]
            if scaled == 'scaled':
               data = scaler.fit_transform(data)
               self.scaled = 'scaled'  # Just to memorize if data is scaled or not

            pca = PCA()
            pca.fit(data)
            principal_components = pca.transform(data)

            self.PC1[group] = principal_components[:, 0]

            self.explained_variance_ratio[group]  = pca.explained_variance_ratio_

        return (self.PC1,self.explained_variance_ratio)


    def plot_Pca(self, group, dates):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        #try catch group
        plt.plot(dates,self.PC1[group])
        n = 35  # Display every 10th label
        plt.xticks(dates[::n], rotation=45, fontsize=8)
        plt.title(f'Plot of PC 1: {group} _ {self.scaled}')
        plt.ylabel(' PC1 time series')

        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(self.explained_variance_ratio[group]) + 1), self.explained_variance_ratio[group], marker='o', linestyle='-')
        plt.title(f'Explained Variance Ratio: {group} _ {self.scaled}')
        plt.xlabel('Number of Components')
        plt.ylabel('Ratio')
        plt.grid(True)



'''
Thsi class takes as inpout the macro data, uses the Group_macro_pca class to get the pca of each group, apply the hamilton 
filter class on each time series of PC1 group and get the smoothed regime prob for training_val data and filtered probablities 
of regimes  for the test set.
'''

class Macro_group_probs:
    def __init__(self, path_val_tr, path_test, all_groups = True):
        
        data_tr = pd.read_csv(path_val_tr)
        data_test = pd.read_csv(path_test)

        group_data_tr= Group_macro_Pca(data_tr)
        group_data_test = Group_macro_Pca(data_test)
        
        data_pca_tr, _ = group_data_tr.get_Pca('scaled')
        data_pca_test, _ = group_data_test.get_Pca('scaled')
        
        # If must use the 9 PC1 times series in a multivariate hamilton filter create data with the 9 time series goups
        if all_groups:
            all_tr = []
            all_test = []
            for gr in data_pca_tr.keys():
                
              if gr != 'all_vars': # all vars means all data so it s not a groupe.
                all_tr.append(data_pca_tr[gr])
                all_test.append(data_pca_test[gr])
                
            all_tr = np.array(all_tr)
            all_test = np.array(all_test)
            
            data_pca_tr['all_groups'] = all_tr
            data_pca_test['all_groups'] = all_test

        self.prob_tr = {}
        self.prob_test = {}

        for groupe in data_pca_tr.keys():
            
            model = HamiltonKimModel(data_pca_tr[groupe]) 
            model.run() # run hamilton filter for each groupe of PC1 and also for the 9 PC1 of all the groups
            
            results_tr = model.get_sp_hf() 
            smoothed_prob_tr = results_tr['smoothed_prob'] # smoothed prob for train valid data
            smoothed_prob_tr = np.concatenate(([smoothed_prob_tr[:, 0][0]], smoothed_prob_tr[:, 0])) # the filter doesnt yild the first prob so we make it equal as the second
            
            results_test= model.hamilton_filter(param=None,new_obs=data_pca_test[groupe]) # get filterd probabilites for test set macro
            filtered_prob_test= results_test['fp'][:, 0]
            #####filtered_prob_test = np.concatenate(([filtered_prob_test[:, 0][0]], filtered_prob_test[:, 0]))
            
            self.prob_tr[groupe] = smoothed_prob_tr
            self.prob_test[groupe] = filtered_prob_test
            

    def write_to_file(self, folder_path):
        with open(folder_path+'/macro_tr_prob.pkl', 'wb') as file:
            pickle.dump(self.prob_tr, file)
        with open(folder_path+'/macro_test_prob.pkl', 'wb') as file:
            pickle.dump(self.prob_test, file)

    def get_group_prob(self):
        return self.prob_tr,self.prob_test
    

'''
this function run group data run pca on groupes and run hamilton filter on each group and all groups,
it get macro prob associated with each case
'''
        
def run_Hamilton_filter():
    macro_tr = np.load('datasets/macro/macro_train.npz')
    macro_val = np.load('datasets/macro/macro_valid.npz')
    macro_test = np.load('datasets/macro/macro_test.npz')
    
    
    df_train = pd.DataFrame(data=macro_tr['data'], index=macro_tr['date'], columns=macro_tr['variable'])
    df_train.index.name = 'Date'
    
    df_val = pd.DataFrame(data=macro_val['data'], index=macro_val['date'], columns=macro_val['variable'])
    df_val.index.name = 'Date'
    
    test_macro = pd.DataFrame(data=macro_test['data'], index=macro_test['date'], columns=macro_test['variable'])
    test_macro.index.name = 'Date'
    
    train_val_macro = pd.concat([df_train, df_val], axis=0)
    train_val_macro.to_csv('datasets/train_val_macro.csv')
    test_macro.to_csv('datasets/test_macro.csv')
    
    
    
    tr_val_macro_path = 'datasets/train_val_macro.csv'
    test_macro_path = 'datasets/test_macro.csv'
    
    hamilton = Macro_group_probs(tr_val_macro_path,test_macro_path, all_groups= True)
    prob_tr_val, prob_test = hamilton.get_group_prob()
    hamilton.write_to_file('macro_probabilities') 
    
    
   
    
#to write prob of files run : run_Hamilton_filter()   