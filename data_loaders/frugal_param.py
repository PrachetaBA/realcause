"""
Data Loader for the Frugal DGP datasets.

"""

# Import libraries
import sys
import pandas as pd
from consts import BASE_DATASETS_FOLDER

sys.path.append(BASE_DATASETS_FOLDER)


def load_frugal_dgp(identifier, data_format='pandas'):
    """
    Function to load the Frugal DGP datasets, that have been generated
    using the R package `causl`.

    The possible dataset identifiers are:
    1. "dgp1":
        - 5k samples
        - 3 covariates
    2. "dgp1_unobs":
        - 5k samples
        - 3 covariates (in total)
        - 1 unobserved confounder
    3. "dgp2":
        - 5k samples
        - 4 covariates; similar to M1 model in FrugalFlows paper
    4. "dgp2_unobs":
        - 5k samples
        - 4 covariates (in total)
        - 2 unobserved confounder
    5. "dgp3":
        - 5k samples
        - 10 covariates; similar to M3 model in FrugalFlows paper
    6. "dgp3_unobs":
        - 5k samples
        - 10 covariates (in total)
        - 3 unobserved confounders
    7. "dgp4":
        - 5k samples
        - 2 covariates
    8. "dgp4_unobs":
        - 5k samples
        - 2 covariates (in total)
        - 1 unobserved confounder
    9. ....
    """
    # By default, there is no unobserved confounder
    observed_cov = []
    unobserved_cov = []

    if identifier in ['dgp1', 'dgp1_unobs', 'dgp1_conf']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp1.csv')
        true_ate = 5.0
        continuous_vars = ['X1', 'X2', 'X3']
        categorical_vars = []
        if 'unobs' in identifier:
            observed_cov = ['X1', 'X3']
            unobserved_cov = ['X2']
    elif identifier in ['dgp1_small', 'dgp1_small_unobs']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp1_small.csv')
        true_ate = 5.0
        continuous_vars = ['X1', 'X2', 'X3']
        categorical_vars = []
        if 'unobs' in identifier:
            observed_cov = ['X1', 'X3']
            unobserved_cov = ['X2']
    elif identifier in ['dgp2', 'dgp2_unobs']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp2.csv')
        true_ate = 5.0
        continuous_vars = ['X1', 'X2', 'X3', 'X4']
        categorical_vars = []
        if 'unobs' in identifier:
            observed_cov = ['X1', 'X2', 'X3']
            unobserved_cov = ['X4']
    elif identifier in ['dgp2_small', 'dgp2_small_unobs']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp2_small.csv')
        true_ate = 5.0
        continuous_vars = ['X1', 'X2', 'X3', 'X4']
        categorical_vars = []
        if 'unobs' in identifier:
            observed_cov = ['X1', 'X2', 'X3']
            unobserved_cov = ['X4']
    elif identifier in ['dgp3', 'dgp3_unobs']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp3.csv')
        true_ate = -5.0
        continuous_vars = ['X1', 'X2', 'X3', 'X4', 'X5']
        categorical_vars = ['X6', 'X7', 'X8', 'X9', 'X10']
        if 'unobs' in identifier:
            observed_cov = ['X1', 'X2', 'X4', 'X5', 'X6', 'X8', 'X9', 'X10']
            unobserved_cov = ['X3', 'X7']
    elif identifier in ['dgp3_small', 'dgp3_small_unobs']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp3_small.csv')
        true_ate = -5.0
        continuous_vars = ['X1', 'X2', 'X3', 'X4', 'X5']
        categorical_vars = ['X6', 'X7', 'X8', 'X9', 'X10']
        if 'unobs' in identifier:
            observed_cov = ['X1', 'X2', 'X4', 'X5', 'X6', 'X8', 'X9', 'X10']
            unobserved_cov = ['X3', 'X7']
    elif identifier in ['dgp4', 'dgp4_unobs']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_dgp4.csv')
        true_ate = -2.0
        continuous_vars = ['X1', 'X2']
        categorical_vars = []
        if 'unobs' in identifier:
            observed_cov = ['X1']
            unobserved_cov = ['X2']
    elif identifier in ['dgp5', 'dgp6']:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/frugal_param/frugal_param_{identifier}.csv')
        if identifier == 'dgp5':
            true_ate = -10.0
        elif identifier == 'dgp6':
            true_ate = -1.5
        continuous_vars = ['X1', 'X2', 'X3', 'X4', 'X5']
        categorical_vars = ['X6', 'X7', 'X8', 'X9', 'X10']
    else:
        raise ValueError(f'Unknown identifier: {identifier}. ')
    if 'unobs' in identifier:
        df_info = {
            'outcome_col': 'Y',
            'treatment_col': 'T',
            'continuous_vars': continuous_vars,
            'categorical_vars': categorical_vars,
            'observed_cov': observed_cov,
            'unobserved_cov': unobserved_cov,
            'true_ate': true_ate,
            'sample_size': df.shape[0],
        }
    else:
        df_info = {
            'outcome_col': 'Y',
            'treatment_col': 'T',
            'continuous_vars': continuous_vars,
            'categorical_vars': categorical_vars,
            'observed_cov': continuous_vars + categorical_vars,
            'unobserved_cov': unobserved_cov,
            'true_ate': true_ate,
            'sample_size': df.shape[0],
        }
    if data_format == 'numpy':
        d = {
            'w': df[continuous_vars + categorical_vars].to_numpy(),
            't': df['T'].to_numpy(),
            'y': df['Y'].to_numpy(),
        }
    elif data_format == 'pandas':
        d = {
            'w': df[continuous_vars + categorical_vars],
            't': df['T'],
            'y': df['Y'],
        }
    else:
        raise ValueError(f'Unknown data format: {data_format}. ')
    return d, df_info


if __name__ == '__main__':
    d, df_info = load_frugal_dgp(identifier='dgp3', data_format='numpy')
    w, t, y = d['w'], d['t'], d['y']
    print(df_info['true_ate'])
    print(df_info['sample_size'])
