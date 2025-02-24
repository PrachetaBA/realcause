"""Data loader for the simple linear datasets that we have generated 
for comparing the experiments in our paper. 

The dataset generation is located in the `notebooks/synthetic_datasets.ipynb`
"""

# Import libraries
import pandas as pd
from consts import NFL_CE_BASE_DATASETS_FOLDER

def get_synthetic_linear_data(dataset_id,
                              data_format='numpy',
                              return_ites=True,
                              return_counterfactual_outcomes=False):
    """Load the synthetic linear dataset with the given dataset_id."""
    if dataset_id in ['dgp1', 'dgp2']:
        df = pd.read_csv(f'{NFL_CE_BASE_DATASETS_FOLDER}/synthetic/syn_linear_{dataset_id}.csv')
    else:
        raise ValueError(f"Invalid dataset_id: {dataset_id}")
    
    if data_format == 'numpy': 
        d = {
            'w': df.drop(columns=['y', 'y_cf','y1','y0','ite'], axis='columns').to_numpy(),
            't': df['t'].to_numpy(),
            'y': df['y'].to_numpy(),
        }
    elif data_format == 'pandas':
        d = {
            'w': df.drop(columns=['y', 'y_cf','y1','y0','ite'], axis='columns'),
            't': df['t'],
            'y': df['y'],
        }
    else:
        raise ValueError(f"Invalid data_format: {data_format}")
    
    if return_ites:
        d['ites'] = df['ite'].to_numpy() if data_format == 'numpy' else df['ite']
        
    if return_counterfactual_outcomes:
        d['counterfactual_outcomes_0'] = df['y0'].to_numpy() if data_format == 'numpy' else df['y0']
        d['counterfactual_outcomes_1'] = df['y1'].to_numpy() if data_format == 'numpy' else df['y1']
        
    return d