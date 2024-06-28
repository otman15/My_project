
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle



'''
Hamilton class:
  Prend des séries temporelles univariées ou multivariées, applique un filtre de Hamilton et un lissage de Kim
et calcule les régimes de probabilité filtrés et lissés.
si les données sont utilisées pour calculer les paramètres du filtre (données d'entraînement), le lissage de Kim est utilisé car
on suppose que nous avons accès à toutes les données.

  Si de nouvelles données de test sont données pour prédire de nouveaux régimes de probabilités, nous supposons que nous n'avons pas
accès à toute la série temporelle mais seulement aux observations présentes, donc le lissage de Kim n'est pas
appliqué mais les paramètres entraînés et la dernière probabilité sont utilisés pour inférer de nouvelles probabilités.

  Le code dans la classe est basé sur : 
        Handbook of Econometrics, Volume 1 V, 
    édité par R.F. Engle et D.L. McFadden, chapitre 50 State Space Models, 
    4 Variables d'état à valeurs discrètes.
    
      Hamilton 1989: A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle: section 4 
     James D. Hamilton : Regime-Switching Models 2005

Il est également basé sur le code Programs for estimation of Markov switching models using the EM algorithm.
sur la page web : https://econweb.ucsd.edu/~jhamilto/software.htm#Markov

    
'''



class HamiltonKimModel:
    def __init__(self, data, retard=True): # retard = lag

        if data.ndim == 1:  # # Série temporelle univariée
            data = data[np.newaxis, :]  # Convertir en tableau 2D avec une seule ligne
        self.retard = retard
        self.y = data[:, 1:]  # Inclure tous les éléments sauf le premier pour chaque série
        if self.retard:
            self.x = data[:, :-1]  # Inclure tous les éléments sauf le dernier pour chaque série
        self.param = None
    
    
    
    def filtering(self, param):

        n_series = self.y.shape[0]

        if self.retard:
            const,beta,sigma, p,q = self.reshape_params(param, n_series)
        else: const,sigma, p,q = self.reshape_params(param, n_series)
                   
        nobs = self.y.shape[1]
        
        rho = (1 - p) / (2 - q - p)
        S = np.array([rho, 1 - rho])
        

        #  probabilité filtrée
        fp = np.zeros((nobs, 2))

        loglike = 0
        for t in range(nobs):
            # État précédent
            S_prev = [S[1],S[0]]
            # Estimation de régression
            mu0 = np.zeros(n_series)
            mu1 = np.zeros(n_series)
            if self.retard:
                    mu0 = [const[n, 0] + beta[n, 0] * self.x[n, t] for n in range(n_series)]
                    mu1 = [const[n, 1] + beta[n, 1] * self.x[n, t] for n in range(n_series)]

            else:
                    mu0 = const[:, 0]
                    mu1 = const[:, 1]

            # Densités sous les deux régimes à t
            fy_g_st0 = 1
            fy_g_st1 = 1
            for n in range(n_series):# Par suppposition : Observations indépendantes f(y1, y2, ...) = f(y1) * f(y2) * ...
                fy_g_st0 *= (1 / np.sqrt(2 * np.pi * sigma[n, 0] ** 2)) * np.exp(-((self.y[n, t] - mu0[n]) ** 2) / (2 * sigma[n, 0] ** 2))
                fy_g_st1 *= (1 / np.sqrt(2 * np.pi * sigma[n, 1] ** 2)) * np.exp(-((self.y[n, t] - mu1[n]) ** 2) / (2 * sigma[n, 1] ** 2))
                
            p1 = [S_prev[0] * p * fy_g_st0, S_prev[0] * (1 - p) * fy_g_st1,
                   S_prev[1] * (1 - q) * fy_g_st0,S_prev[1] * q * fy_g_st1]
           
            # Densité yt
            f_yt = p1[0]+ p1[1]+ p1[2]+ p1[3]

            if f_yt < 0 or np.isnan(f_yt):
                loglike = -100000000
                break
            
            # État à t+1 étant donné t
            S[1] = (p1[0]+p1[2]) / f_yt
            S[0] = (p1[1]+ p1[3]) / f_yt

            fp[t, :] = [S[1], S[0]]  

            loglike += np.log(f_yt)

        return {'loglike': -loglike, 'fp': fp}
    
    def reshape_params(self, param, n_series):

        
        if self.retard:
            const = param[:n_series*2].reshape(n_series, 2)
            beta = param[n_series*2:n_series*4].reshape(n_series, 2)
            sigma = param[n_series*4:n_series*6].reshape(n_series, 2)
            p7, p8 = param[-2:]
            p = 1 / (1 + np.exp(-p7))
            q = 1 / (1 + np.exp(-p8))
            return const, beta, sigma, p, q
        else:
            const = param[:n_series*2].reshape(n_series, 2)
            sigma = param[n_series*2:n_series*4].reshape(n_series, 2)
            p7, p8 = param[-2:]
            p = 1 / (1 + np.exp(-p7))
            q = 1 / (1 + np.exp(-p8))
            return const, sigma, p,q
   
    def predicting(self, new_obs):# Appliquer le filtre une fois pour obtenir les probabilités filtrées
        
        param = self.param
        n_series = self.y.shape[0]
        initial_probs = self.sp[-1]
        new_y = new_obs if new_obs.ndim > 1 else new_obs[np.newaxis, :]
        last_y = self.y[:, -1].reshape(n_series, 1)
        combined_y = np.hstack((last_y, new_y))
        if self.retard:
            self.x = combined_y[:, :-1]  
        self.y = combined_y[:, 1:] 
        E0_t, E1_t = initial_probs
        
        if self.retard:
            const,beta,sigma, p,q = self.reshape_params(param, n_series)
        else: const,sigma, p,q = self.reshape_params(param, n_series)
            
        
        nobs = self.y.shape[1]
       
        rho = (1 - p) / (2 - q - p)
        S = np.array([rho, 1 - rho])
        
        fp = np.zeros((nobs, 2))
 
 
        for t in range(nobs):

            S_prev = [S[1],S[0]]

            mu0 = np.zeros(n_series)
            mu1 = np.zeros(n_series)
            if self.retard:
                    mu0 = [const[n, 0] + beta[n, 0] * self.x[n, t] for n in range(n_series)]
                    mu1 = [const[n, 1] + beta[n, 1] * self.x[n, t] for n in range(n_series)]
 
            else:
                    mu0 = const[:, 0]
                    mu1 = const[:, 1]

            fy_g_st0 = 1
            fy_g_st1 = 1
            for n in range(n_series):
                fy_g_st0 *= (1 / np.sqrt(2 * np.pi * sigma[n, 0] ** 2)) * np.exp(-((self.y[n, t] - mu0[n]) ** 2) / (2 * sigma[n, 0] ** 2))
                fy_g_st1 *= (1 / np.sqrt(2 * np.pi * sigma[n, 1] ** 2)) * np.exp(-((self.y[n, t] - mu1[n]) ** 2) / (2 * sigma[n, 1] ** 2))
                
            p1 = [S_prev[0] * p * fy_g_st0, S_prev[0] * (1 - p) * fy_g_st1,
                   S_prev[1] * (1 - q) * fy_g_st0,S_prev[1] * q * fy_g_st1]

            f_yt = p1[0]+ p1[1]+ p1[2]+ p1[3]

            S[1] = (p1[0]+p1[2]) / f_yt
            S[0] = (p1[1]+ p1[3]) / f_yt
 
            fp[t, :] = [S[1], S[0]]  #

 
        return {'fp': fp}
        
        
        
        

                   

    def smoothing(self): # obtenir les probabilités lissées
        p7, p8 = self.param[-2:]

        nobs = self.y.shape[1]

        p = 1 / (1 + np.exp(-p7))
        q = 1 / (1 + np.exp(-p8))

        sp = np.zeros((nobs, 2))

        # obtenir les estimations filtrées
        hf = self.hf

        fp = hf['fp']  

        T = nobs
        sp[T - 1, :] = fp[T - 1, :]

        # Obtenir les sorties filtrées
        for is_ in range(T - 2, -1, -1):
            p1 = (sp[is_ + 1, 0] * fp[is_, 0] * p) / (fp[is_, 0] * p + fp[is_, 1] * (1 - q))
            p2 = (sp[is_ + 1, 1] * fp[is_, 0] * (1 - p)) / (fp[is_, 0] * (1 - p) + fp[is_, 1] * q)
            p3 = (sp[is_ + 1, 0] * fp[is_, 1] * (1 - q)) / (fp[is_, 0] * p + fp[is_, 1] * (1 - q))
            p4 = (sp[is_ + 1, 1] * fp[is_, 1] * q) / (fp[is_, 0] * (1 - p) + fp[is_, 1] * q)

            sp[is_, 0] = p1 + p2
            sp[is_, 1] = p3 + p4

        return sp

    def fit_markov_switching_model(self):
        if self.retard:
            n_series = self.y.shape[0]
            init_guess = np.ones(n_series * 6 + 2)
            init_guess[-2:] = [3.0, 2.9]
            bounds = [(None, None)] * (n_series * 6) + [(1e-6, None), (1e-6, None)]
        else:
            n_series = self.y.shape[0]
            init_guess = np.ones(n_series * 4 + 2)
            init_guess[-2:] = [3.0, 2.9]
            bounds = [(None, None)] * (n_series * 4) + [(1e-6, None), (1e-6, None)]


        mle = minimize(self.likelihood_function, init_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50000})
        return mle.x

    def likelihood_function(self, param):
        result = self.filtering(param=param)
        return result['loglike']

    def run(self):
        self.param = self.fit_markov_switching_model()
        self.hf = self.filtering(self.param)
        self.sp = self.smoothing()


    def get_sp_hf(self):# résultats: probabilité lissée et probabilité filtrée
        return {'smoothed_prob':self.sp, 'filtered_prob': self.hf['fp']}


    def plot_results(self, dates= None):
        plt.figure(figsize=(8, 6))

        if not dates:
            plt.plot(self.sp[:, 0], label='Regime 1', color='green', linewidth=3, linestyle='-')
            plt.plot(self.sp[:, 1], label='Regime 2', color='blue', linewidth=3, linestyle='-')
        else:
            plt.plot(dates,self.sp[:, 0], label='Regime 1', color='green', linewidth=3, linestyle='-')
            plt.plot(dates,self.sp[:, 1], label='Regime 2', color='blue', linewidth=3, linestyle='-')
            n = 35 # Afficher chaque 10e étiquette
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
   Cette classe est utilisée pour regrouper les données macro par groupe : 10 groupes incluant
un groupe contenant toutes les données (sans regroupement). 
  Elle effectue une ACP sur chaque groupe et retourne le première composante de chaque groupe
  macro. Elle produit un dictionnaire de la forme (groupe : PC1).
  Elle peut également récupérer les données et tracer PC1.

  J'utilise ici les mêmes données que celles utilisées par Chen et Al. (2023), 
disponibles sur FRED-MD: A Monthly Database for Macroeconomic Research.
 Pour le regroupement, j'utilise le même regroupement que celui décrit dans l'article https://research.stlouisfed.org/wp/2015/2015-012.pdf
'''


class Group_macro_Pca:

    def __init__(self, data1):

        self.data1 = data1
        self.scaled = 'non scaled'
        self.data_dict = self.group()

        self.groups = ['Production_Revenus', 'Marche_de_travail', 'Logement','Conso_Ordres' , 'Monnaie_credit',
                       'Taux_interet_change', 'Prix', 'Marches_Boursiers','Cross_sec_med_char',
                       'Welch_indicators','Toutes_vars']

    def group(self):
        #group1 = data1.columns[1:20]
        Production_Revenus1 = ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS',  'IPFINAL','IPCONGD', 'IPDCONGD', 'IPNCONGD', 
                        'IPBUSEQ',  'IPMAT','IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S',  'IPFUELS', 'CUMFNS']

        #group2_names = data1.columns[20:48]
        Marche_de_travail2 = ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN',
               'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx',
               'PAYEMS', 'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP',
               'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE',
               'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008','CES2000000008','CES3000000008']


        #group3_names = data1.columns[48:58] # 49 ....  61

        Logement3 = ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE',
               'PERMITMW', 'PERMITS', 'PERMITW']

        #group4 = data1.columns[3:6]

        Conso_Ordres4 = ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'AMDMNOx', 'AMDMUOx', 'BUSINVx', 'ISRATIOx']

        #group5 = data1.columns[62:72]

        Monnaie_credit5 = ['M1SL', 'M2SL', 'M2REAL', 'AMBSL', 'TOTRESNS', 'NONBORRES', 'BUSLOANS',
               'REALLN', 'NONREVSL', 'CONSPI', 'MZMSL', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST']

        #groupr6  = data1.columns[76:97]

        Taux_interet_change = ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA',
               'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM',
               'T10YFFM', 'AAAFFM', 'BAAFFM', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx',
               'EXCAUSx']

        #groupr7  = data1.columns[97:117]

        Prix7 = ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM',
               'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC',
               'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2',
               'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA',
               'DSERRG3M086SBEA']

        #group8 = data1.columns[72:76]

        Marches_Boursiers8 = ['S&P 500', 'S&P: indust', 'S&P div yield', 'S&P PE ratio','VXOCLSx']
        
        Cross_sec_med_char = [
                    'A2ME', 'AC', 'AT', 'ATO', 'BEME', 'Beta', 'C', 'CF', 'CF2P', 'CTO', 'D2A', 'D2P',
                    'DPI2A', 'E2P', 'FC2Y', 'IdioVol', 'Investment', 'Lev', 'LME', 'LT_Rev', 'LTurnover', 
                    'MktBeta', 'NI', 'NOA', 'OA', 'OL', 'OP', 'PCM', 'PM', 'PROF', 'Q', 'r2_1', 'r12_2', 
                    'r12_7', 'r36_13', 'Rel2High', 'Resid_Var', 'RNA', 'ROA', 'ROE', 'S2P', 'SGA2S', 
                    'Spread', 'ST_REV', 'SUV', 'Variance'
                ]

        Welch_indicators = [ 'dp', 'ep', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']



        self.data_dict = {
            'Production_Revenus': self.data1[Production_Revenus1], #'Production_Revenus': self.data1[Production_Revenus1].values
            'Marche_de_travail': self.data1[Marche_de_travail2],
            'Logement': self.data1[Logement3],
            'Conso_Ordres': self.data1[Conso_Ordres4],
            'Monnaie_credit': self.data1[Monnaie_credit5],
            'Taux_interet_change': self.data1[Taux_interet_change],
            'Prix': self.data1[Prix7],
            'Marches_Boursiers': self.data1[Marches_Boursiers8],
            'Cross_sec_med_char': self.data1[Cross_sec_med_char],
            'Welch_indicators': self.data1[Welch_indicators],
            'Toutes_vars': self.data1.drop(columns=Cross_sec_med_char).iloc[:,1:] # ne pas inclure les chars

        }


    def get_data(self, variable_name=None):
      
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
        
        if groups == None : groups = self.groups
        self.group()

        for group in groups:

            scaler = StandardScaler()
            data = self.data_dict[group]
            if scaled == 'scaled':
               data = scaler.fit_transform(data)
               self.scaled = 'scaled'  

            pca = PCA()
            pca.fit(data)
            principal_components = pca.transform(data)

            self.PC1[group] = principal_components[:, 0]

            self.explained_variance_ratio[group]  = pca.explained_variance_ratio_

        return (self.PC1,self.explained_variance_ratio)


    def plot_Pca(self, group, dates):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        
        plt.plot(dates,self.PC1[group])
        n = 35  
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
  Cette classe prend en entrée les données macroéconomiques, utilise la classe Group_macro_pca
pour obtenir l'ACP de chaque groupe, applique la classe Hamilton Filter à chaque série temporelle
du groupe PC1, puis obtient les probabilités de régime lissées pour les données 
d'entraînement/validation et les probabilités filtrées de régime pour l'ensemble de test.
'''

class Macro_group_probs:
    def __init__(self, path_val_tr, path_test, Tous_groupes = True, retard='avec_retard'):
        
        is_retard = retard=='avec_retard'
        self.retard = retard
        data_tr = pd.read_csv(path_val_tr)
        data_test = pd.read_csv(path_test)

        group_data_tr= Group_macro_Pca(data_tr)
        group_data_test = Group_macro_Pca(data_test)
        
        data_pca_tr, _ = group_data_tr.get_Pca('scaled')
        data_pca_test, _ = group_data_test.get_Pca('scaled')
        
        # Si vous devez utiliser toutes les 9 séries temporelles de PC1 dans le filtre de Hamilton multivarié, créez une structure de données avec les 9 groupes de séries temporelles.
        if Tous_groupes:
            all_tr = []
            all_test = []
            for gr in data_pca_tr.keys():
                
              if (gr != 'Toutes_vars') and (gr != 'Cross_sec_med_char'): 
                all_tr.append(data_pca_tr[gr])
                all_test.append(data_pca_test[gr])
                
            all_tr = np.array(all_tr)
            all_test = np.array(all_test)
            
            data_pca_tr['Tous_groupes'] = all_tr
            data_pca_test['Tous_groupes'] = all_test

        self.prob_tr = {}
        self.prob_test = {}

        for groupe in data_pca_tr.keys():
            
            model = HamiltonKimModel(data_pca_tr[groupe], is_retard) 
            model.run() # Exécutez le filtre de Hamilton pour chaque groupe de PC1 ainsi que pour les 9 PC1 de l'ensemble des groupes.
            
            results_tr = model.get_sp_hf() 
            smoothed_prob_tr = results_tr['smoothed_prob'] 
            smoothed_prob_tr = np.concatenate(([smoothed_prob_tr[:, 0][0]], smoothed_prob_tr[:, 0])) # Le filtre ne produit pas la première probabilité, je la fixe égale à la deuxième.
            
            results_test= model.predicting(new_obs=data_pca_test[groupe]) 
            filtered_prob_test= results_test['fp'][:, 0]
            
            self.prob_tr[groupe] = smoothed_prob_tr
            self.prob_test[groupe] = filtered_prob_test
            

    def write_to_file(self, folder_path):
        with open(folder_path+'/macro_tr_prob_' + self.retard + '.pkl', 'wb') as file:
            pickle.dump(self.prob_tr, file)
        with open(folder_path+'/macro_test_prob_'+ self.retard +'.pkl', 'wb') as file:
            pickle.dump(self.prob_test, file)

    def get_group_prob(self):
        return self.prob_tr,self.prob_test
    

'''
  Cette fonction regroupe les données, applique l'ACP sur les groupes, exécute le filtre de Hamilton
sur chaque groupe ainsi que sur l'ensemble des groupes, et obtient les probabilités de régime associées
à chaque cas.
'''
        
def run_Hamilton_filter(retard='sans_retard'):
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
    
    hamilton = Macro_group_probs(tr_val_macro_path,test_macro_path, Tous_groupes= True, retard = retard)
    prob_tr_val, prob_test = hamilton.get_group_prob()
    hamilton.write_to_file('macro_probabilities') 

    
    
   
  
#Pour écrire les probabilités dans des fichiers, exécutez la fonction `run_Hamilton_filter()`
#run_Hamilton_filter('avec_retard') 
#run_Hamilton_filter('sans_retard') 