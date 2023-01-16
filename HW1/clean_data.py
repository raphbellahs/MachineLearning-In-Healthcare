# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = pd.DataFrame([pd.to_numeric(CTG_features[ctg], errors='coerce') for ctg in CTG_features]).T.drop(columns = [extra_feature]).dropna()
    
    # --------------------------------------------------------------------------
    return c_ctg

def rand_sampling(x, feature):
    """
    :param x : Panda dataframe 
    :param feature : A feature to remplace with sample of the same column
    :return : Panda dataframe with the column replaced
    
    """
    if x == np.nan:
        while(x == np.nan):
            rand_idx = np.random.choice(len(feature))
            x = feature.iloc[rand_idx]
    return x 


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
#     c_cdf = pd.DataFrame([pd.to_numeric(CTG_features[ctg], errors='coerce') for ctg in CTG_features]).T.drop(columns = [extra_feature]
#     print("NEW TEST")
#     for col in CTG_features.columns:
#         if col != "MSTV":
#             continue
#         for i in range(len(CTG_features[col])):
# #             print(CTG_features[col].loc[i])
# #             print(pd.isna(CTG_features[col].loc[i]))
#             while pd.isna(CTG_features[col].loc[i]):
                
#                 # print(temp[col].iloc[i])
#                 rand_idx = np.random.choice(len(CTG_features[col]))
#                 x = CTG_features[col].iloc[rand_idx]
#                 # print(f"Random Index : {rand_idx} and value : {x}")
#                 # prev = temp[col].iloc[i]
#                 CTG_features[col].loc[i] = x
                    
#     # -------------------------------------------------------------------------

    c_cdf = pd.DataFrame({feature: pd.to_numeric(CTG_features[feature], errors='coerce') for feature in CTG_features if
                          feature != extra_feature})
    for column in c_cdf:
        for i, value in enumerate(c_cdf[column]): 
            if value != value:
                rand_idx = np.random.choice(len(c_cdf[column]))
                x = c_cdf.loc[rand_idx, column]
                while x != x: 
                    rand_idx = np.random.choice(len(column))
                    x = c_cdf.loc[rand_idx, column]
                c_cdf[column][i] = x
    return c_cdf


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for column in c_feat:
        dict_column = {}
        dict_column["min"] = c_feat[column].min()
        dict_column["Q1"] = c_feat[column].quantile(0.25)
        dict_column["median"] = c_feat[column].median()
        dict_column["Q3"] = c_feat[column].quantile(0.75)
        dict_column["max"] = c_feat[column].max()
        
        d_summary[column] = dict_column
        
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed
    """
    c_no_outlier = c_feat.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    "The interquartile range (IQR) is computed by the different between the 75th and 25th percentiles."
    for column in c_feat:
        IQR = d_summary[column]['Q3'] - d_summary[column]['Q1']
        Left = d_summary[column]['Q1'] - 1.5*IQR
        Right = d_summary[column]['Q3'] + 1.5*IQR
        c_no_outlier[column] = np.where((c_no_outlier[column] < Left) | (c_no_outlier[column] > Right),np.nan,c_no_outlier[column] )
        # c_no_outlier[column].mask(c_no_outlier[column] > Right or c_no_outlier[column] < Left  ,np.nan , inplace=True)
         
    # -------------------------------------------------------------------------
    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """

    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_samp.drop(c_samp[c_samp[feature] > thresh].index)[feature]
    # -------------------------------------------------------------------------
    return np.array(filt_feature)


class NSD:

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False
    
    def fit(self, CTG_features):
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        
        self.max = CTG_features.apply(np.max)
        self.min = CTG_features.apply(np.min)
        self.mean = CTG_features.apply(np.mean)
        self.std = CTG_features.apply(np.std)
        # -------------------------------------------------------------------------
        self.fit_called = True

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: A two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether or not plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            # ------------------ IMPLEMENT YOUR CODE HERE (for the remaining 3 methods using elif):-----------------------------
            ''' We want X-mu/std for each column '''
            if mode == 'standard':
                for column in ctg_features:
                    ctg_features[column] = ctg_features[column].apply(lambda x: (x - self.mean[column])/self.std[column])
                    nsd_res = ctg_features
                    x_lbl = 'Standardized values [N.U.]'
            
            ''' We want X-min/max-min for each column '''
            if mode == 'MinMax':
                for column in ctg_features:
                    ctg_features[column] = ctg_features[column].apply(lambda x: (x - self.min[column])/(self.max[column] - self.min[column]))
                    nsd_res = ctg_features
                    x_lbl = 'Normalized values [N.U.]'
                    
                  
            ''' We want X-min/max-min for each column '''
            if mode == 'mean':
                for column in ctg_features:
                    ctg_features[column] = ctg_features[column].apply(lambda x: (x - self.mean[column])/(self.max[column] - self.min[column]))
                    nsd_res = ctg_features
                    x_lbl = 'Mean normalized values [N.U.]'
                    
            
                                                                    

            # -------------------------------------------------------------------------
            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == 'none':
            bins = 50
        else:
            bins = 80
            # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        x1 = nsd_res[selected_feat[0]]
        x2 = nsd_res[selected_feat[1]]
        plt.figure(figsize=(10,5))
        plt.hist([x1,x2],label=[selected_feat[0],selected_feat[1]], bins=80)
        plt.legend(borderpad=0.2,fontsize=10)
        plt.show()

            # -------------------------------------------------------------------------

# Debugging block!
if __name__ == '__main__':
    from pathlib import Path
    file = Path.cwd().joinpath(
        'messed_CTG.xls')  # concatenates messed_CTG.xls to the current folder that should be the extracted zip folder
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
                                'Tendency']]
    CTG_morph = CTG_dataset[['CLASS']]
    fetal_state = CTG_dataset[['NSP']]

    extra_feature = 'DR'
    c_ctg = rm_ext_and_nan(CTG_features, extra_feature)