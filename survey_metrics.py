# -*- coding: utf-8 -*-
"""Script to calculate the metrics reported in the paper from the human survey."""

import pandas as pd
import numpy as np
import ranky as rk 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

def calculate_metrics(comp_df, results):
    """Evaluate each univariate stream in a geo_set.

    Args:
        comp_df: pd.DataFrame with the output scores from an algorithm
        results: pd.DataFrame with first 5 cols correspond to ranking from 5 streams in form A 
                    pd.DataFrame with first 5 cols correspond to ranking from 5 streams in form B
    Returns:
        metrics_df: pd.DataFrame 
    """
    output_values_df = []
    for user, user_df in results.iterrows():
        user_a = user_df.iloc[:5]
        user_b = user_df.iloc[5:]
        binary_a = ~user_a.isna()
        binary_b = ~user_b.isna()
        user_a=user_a.fillna(0).values
        user_b=user_b.fillna(0).values
        for alg, alg_df in comp_df.T.iterrows():
            comp_a = alg_df.iloc[:5]
            comp_b = alg_df.iloc[5:]
            if i==0:#0
                binary_comp_a = comp_a
                binary_comp_b = comp_b
            else:
                comp_a = (rk.rank(comp_a.fillna(0), reverse=True)*~comp_a.isna()).replace(0, np.nan)
                comp_b = (rk.rank(comp_b.fillna(0), reverse=True)*~comp_b.isna()).replace(0, np.nan)
                binary_comp_a = comp_a.index.isin(comp_a.nlargest(binary_a.sum()).index)
                binary_comp_b = comp_b.index.isin(comp_b.nlargest(binary_b.sum()).index)
            for name_met, metric in { 'Accuracy':accuracy_score, 'F1':f1_score, 'bal_acc':balanced_accuracy_score}.items():
                output_values_df.append({'user':user, 'alg':alg, 'met':name_met, 'form':'a', 'val':metric(binary_a, binary_comp_a)})
                output_values_df.append({'user':user, 'alg':alg, 'met':name_met, 'form':'b', 'val':metric(binary_b, binary_comp_b)})
            #ROC AUC Separate and only if multiclass:
            if len(pd.unique(binary_a))==2:
                output_values_df.append({'user':user, 'alg':alg, 'met':'ROCAUC', 'form':'a', 'val':roc_auc_score(binary_a, comp_a)})
            if len(pd.unique(binary_b))==2:
                output_values_df.append({'user':user, 'alg':alg, 'met':'ROCAUC', 'form':'b', 'val':roc_auc_score(binary_b, comp_b)})
                output_values_df.append({'user':user, 'alg':alg, 'met':'Correlation', 'form':'a', 'val':rk.corr(user_a, comp_a, method='swap')})
                output_values_df.append({'user':user, 'alg':alg, 'met':'Correlation', 'form':'b', 'val':rk.corr(user_b, comp_b, method='swap')})
                output_values_df.append({'user':user, 'alg':alg, 'met':'Distance', 'form':'a', 'val':rk.dist(user_a, comp_a, method='euclidean')})
                output_values_df.append({'user':user, 'alg':alg, 'met':'Distance', 'form':'b', 'val':rk.dist(user_b, comp_b, method='euclidean')})
    return pd.DataFrame(output_values_df)
