"""Script to load the ACIC 2019 datasets which will then be used to 
generate more synthetic datasets."""

# Import libraries
import pandas as pd
import numpy as np
import consts

def load_low_dim(dataset_identifier, data_format='numpy', return_ites=True):
    """Loads the low dimensional datasets, specified by the dataset_identifier."""
    df = pd.read_csv(f'{consts.ROOT_DIR}/base_datasets/low_dim_{dataset_identifier}.csv')
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
    return d
