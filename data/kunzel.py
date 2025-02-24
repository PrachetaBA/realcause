"""Data loader for the synthetic DGPs used in the prior literature.
Specifically, we use this data loader for the set of DGPs used in the
following papers
1. Kunzel et. al (2019)
2. Kennedy (2020)
3. Curth et al (2021)
"""

# Import libraries
import numpy as np
import pandas as pd
from consts import NFL_BASE_DATASETS_FOLDER

def get_kunzel_data(dataset_id,
                    sample_size,
                    data_format='numpy',
                    return_ites=True,
                    return_counterfactual_outcomes=False):
    """Data loader for the Kunzel et al (2019) synthetic DGP.""" 
    df = pd.read_csv(f'{NFL_BASE_DATASETS_FOLDER}/kunzel_{dataset_id}_ss_{sample_size}_train.csv', index_col=0)
    if data_format == 'numpy':
        d = {
            'w': df.drop(columns=['T', 'Yobs', 'Y1', 'Y0'], axis='columns').to_numpy(),
            't': df['T'].to_numpy(),
            'y': df['Yobs'].to_numpy(),
        }
    elif data_format == 'pandas':
        d = {
            'w': df.drop(columns=['T', 'Yobs', 'Y1', 'Y0'], axis='columns'),
            't': df['T'],
            'y': df['Yobs'],
        }
    else:
        raise ValueError(f"Invalid data format: {data_format}")

    if return_ites:
        ites = np.array(df['Y1'] - df['Y0'])
        d['ites'] = pd.Series(ites) if data_format == 'pandas' else ites

    if return_counterfactual_outcomes:
        d['counterfactual_outcomes_0'] = df['Y0'] if data_format == 'pandas' \
            else df['Y0'].to_numpy()
        d['counterfactual_outcomes_1'] = df['Y1'] if data_format == 'pandas' \
            else df['Y1'].to_numpy()

    return d