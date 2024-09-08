"""Script to load the ACIC 2019 datasets which will then be used to 
generate more synthetic datasets."""

# Import libraries
import pandas as pd
import numpy as np
import consts

def load_low_dim(dataset_identifier, data_format='numpy', return_ites=True, return_counterfactual_outcomes=False):
    """Loads the low dimensional datasets, specified by the dataset_identifier."""
    df = pd.read_csv(f'{consts.ROOT_DIR}/base_datasets/acic19_low_dim_{dataset_identifier}.csv')
    if data_format == 'numpy':
        d = {
            'w': df.drop(['A', 'Y', 'Y_cf'], axis='columns').to_numpy(),
            't': df['A'].to_numpy(),
            'y': df['Y'].to_numpy()
        }
    elif data_format == 'pandas':
        d = {
            'w': df.drop(['A', 'Y', 'Y_cf'], axis='columns'),
            't': df['A'],
            'y': df['Y']
        }
    else:
        raise ValueError('Invalid data_format')

    if return_ites:
        # Compute ITES for dataset
        ites = []
        for _, row in df.iterrows():
            if row['A'] == 1:
                ites.append(row['Y'] - row['Y_cf'])
            else:
                ites.append(row['Y_cf'] - row['Y'])
        ites_np = np.array(ites)
        d['ites'] = pd.Series(ites) if data_format == 'pandas' else ites_np
        
    if return_counterfactual_outcomes:
        # Create two columns 'counterfactual_outcomes_0' and 'counterfactual_outcomes_1'
        # which contains the counterfactual outcomes for A=0 and A=1 respectively
        counterfactual_outcomes_0 = []
        counterfactual_outcomes_1 = []
        for _, row in df.iterrows():
            if row['A'] == 1:
                counterfactual_outcomes_0.append(row['Y_cf'])
                counterfactual_outcomes_1.append(row['Y'])
            else:
                counterfactual_outcomes_0.append(row['Y'])
                counterfactual_outcomes_1.append(row['Y_cf'])
        d['counterfactual_outcomes_0'] = pd.Series(counterfactual_outcomes_0) if data_format == 'pandas' else np.array(counterfactual_outcomes_0)
        d['counterfactual_outcomes_1'] = pd.Series(counterfactual_outcomes_1) if data_format == 'pandas' else np.array(counterfactual_outcomes_1)
        
    return d
