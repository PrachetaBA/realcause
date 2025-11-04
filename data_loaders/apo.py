"""Script to load the APO data (from previous versions of ACIC). This will then be
used to generate observational data with a specific biasing function. This data
will serve as the source dataset for Realcause."""

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
from consts import BASE_DATASETS_FOLDER

sys.path.append(BASE_DATASETS_FOLDER)
from causaleval import biasing_function   # Use the same biasing function as used by FrugalFlow

# Default biasing arguments for the APO dataset
DEFAULT_BIASING_ARGS = {
    'acic_4': {
        'nonlinear': {
            'nl_weight1': 1.0,
            'nl_weight2': -1.0,
            'nl_weight3': 1.0,
            'nl_weight4': 1.0
        }
    },
    'n_acic_4': {
        'linear': {
            'intercept': -1.0,
            'weight': 2.0
        },
        'nonlinear': {
            'nl_weight1': -2.0,
            'nl_weight2': 0.5,
            'nl_weight3': -0.1,
            'nl_weight4': 0.3
        }
    },
    'jdk': {
        'linear': {
            'intercept': -1.0,
            'weight': 2.0
        }
    },
    'postgres': {
        'linear': {
            'intercept': -0.5,
            'weight': -2.0
        },
        'nonlinear': {
            'nl_weight1': -2.0,
            'nl_weight2': 0.5,
            'nl_weight3': -0.1,
            'nl_weight4': 0.3
        }
    }
}

def get_apo_data(identifier,
                 data_format='numpy',
                 return_ites=True,
                 ret_counterfactual_outcomes=False,
                 **kwargs):
    """Function to get an APO dataset that was used in the CausalEval paper.
    
    The possible APO datasets are:
    1. acic_4: ACIC dataset that corresponds to the `apo_acic_4` dataset in the CausalEval paper.
    2. n_acic_4: ACIC dataset that corresponds to the `apo_n_acic_4` dataset in the CausalEval paper (lesser number of covariates)
    3. jdk: APO dataset (corresponding to a real-world experiment)
    
    Keyword arguments:
        - confound_func: Confounding function to be used for the biasing function (default: 'linear').
        if linear:
            - weight: Weight to be used for the biasing function (default: 1).
            - intercept: Intercept to be used for the biasing function (default: 0).
        if nonlinear: 
            - nl_weight1: Weight to be used for the first variable in the nonlinear confounder function (default: 1).
            - nl_weight2: Weight to be used for the second variable in the nonlinear confounder function (default: 1).
            - nl_weight3: Weight to be used for the third variable in the nonlinear confounder function (default: 1).
        - sample_size: Desired sample size for the dataset after subsampling (before doing biasing) (default: 3000).
                       (passing all to sample_size will result in the entire dataset being used)
    
    Returns: 
    - d: A dictionary containing the following information:
        - w: The covariates
        - t: The treatment
        - y: The outcome
        - y0: The counterfactual outcome for the treatment group
        - y1: The counterfactual outcome for the control group
        - ites: The ITEs
        - df_info: A dictionary containing the information about the dataset
    """
    if 'acic' in identifier:
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/causaleval/apo_{identifier}_data.csv')
        # Read in the config
        cfg = pd.read_csv(f'{BASE_DATASETS_FOLDER}/causaleval/apo_{identifier}_config.txt',
                          sep=' ',
                          index_col=None,
                          names=['column', 'type'])
        # If a column exists in the df.columns, but not in cfg['column'], then drop it
        df = df[[
            x for x in df.columns.tolist() if x in
            [*cfg['column'].tolist(), 'counterfactual_outcome_1', 'counterfactual_outcome_0']
        ]]
        
        # Subsample the dataset such that APO is maintained but the sampling is based on the columns specified as in the 
        # index in the cfg file. 
        sample_size = kwargs['sample_size'] if 'sample_size' in kwargs.keys() else 3000
        if sample_size == 'all':
            pass
        else: 
            sample_size = int(sample_size)
            # Extract the unique values of the column named 'index' in the df
            apo_indices = df['index'].unique()
            # Sample the indices such that the sample size is maintained
            sampled_indices = np.random.choice(apo_indices, size=sample_size, replace=False)
            # Subset the dataframe based on the sampled indices
            df = df[df['index'].isin(sampled_indices)]    
        # Drop the index column
        df.drop(columns=['index'], inplace=True)
        
        # Except for the index column, extract all the other columns of type = 'f' in the cfg file
        categorical_var = cfg.loc[cfg['type'] == 'f', 'column'].tolist()
        # Remove the index column from the list
        categorical_var.remove('index')
        # If the following variables are present in categorical_var, then remove them
        one_hot_vars = [
            x for x in categorical_var if x not in
            ['treatment', 'outcome', 'counterfactual_outcome_1', 'counterfactual_outcome_0']
        ]
        df = pd.get_dummies(df, columns=one_hot_vars, drop_first=True, dtype=np.float64)
        # Get the biasing covariate (assume single biasing covariate for now)
        biasing_covariate = cfg.iloc[3]['column']
        # Create an observational dataset from the APO dataset with the desired parameters
        confound_func = kwargs['confound_func'] if 'confound_func' in kwargs.keys() else 'linear'
        if confound_func == 'linear':
            intercept = kwargs['intercept'] if 'intercept' in kwargs.keys() else DEFAULT_BIASING_ARGS[identifier]['linear']['intercept']    # pylint: disable=consider-iterating-dictionary
            weight = kwargs['weight'] if 'weight' in kwargs.keys() else DEFAULT_BIASING_ARGS[identifier]['linear']['weight']    # pylint: disable=consider-iterating-dictionary
            osapo_df = biasing_function.osrct_algorithm(df,
                                                        confound_func_params={
                                                            'para_form': 'linear',
                                                            'intercept': intercept,
                                                            'weight': weight
                                                        },
                                                        treatment_col='treatment',
                                                        confounding_vars=[biasing_covariate])
        elif confound_func == 'nonlinear':
            # Extract the top 3 variables from the cfg file which are type 'n'
            top_3_nvars = cfg.loc[cfg['type'] == 'n', 'column'].tolist()[:3]
            if len(top_3_nvars) < 3:
                raise ValueError(f'Less than 3 nonlinear variables found in the dataset. Found {len(top_3_nvars)} nonlinear variables.')
            osapo_df = biasing_function.osrct_algorithm(df,
                                                        confound_func_params={
                                                            'para_form': 'nonlinear',
                                                            'nl_weight1': kwargs['nl_weight1'] if 'nl_weight1' in kwargs.keys() else DEFAULT_BIASING_ARGS[identifier]['nonlinear']['nl_weight1'],
                                                            'nl_weight2': kwargs['nl_weight2'] if 'nl_weight2' in kwargs.keys() else DEFAULT_BIASING_ARGS[identifier]['nonlinear']['nl_weight2'],
                                                            'nl_weight3': kwargs['nl_weight3'] if 'nl_weight3' in kwargs.keys() else DEFAULT_BIASING_ARGS[identifier]['nonlinear']['nl_weight3'],
                                                            'nl_weight4': kwargs['nl_weight4'] if 'nl_weight4' in kwargs.keys() else DEFAULT_BIASING_ARGS[identifier]['nonlinear']['nl_weight4']
                                                        },
                                                        treatment_col='treatment',
                                                        confounding_vars=top_3_nvars)
        else:
            raise ValueError(f'Confounding function {kwargs["confound_func"]} not recognized.')
        # Ensure we have a standalone DataFrame (avoid SettingWithCopyWarning downstream)
        osapo_df = osapo_df.copy()
        # Compute ITE, expected difference between counterfactual outcomes, after subsampling
        osapo_df.loc[:, 'ite'] = (
            osapo_df['counterfactual_outcome_1'] - osapo_df['counterfactual_outcome_0']
        )
        # Compute ATE, mean of the ITE after subsampling
        ate = osapo_df['ite'].mean()
        # Compute the naive ATE
        naive_ate = osapo_df.loc[osapo_df['treatment'] == 1,
                                 'outcome'].mean() - osapo_df.loc[osapo_df['treatment'] == 0,
                                                                  'outcome'].mean()
        df_info = {
            'outcome_col': 'outcome',
            'treatment_col': 'treatment',
            'true_ate': ate,
            'naive_ate': naive_ate,
            'sample_size': osapo_df.shape[0]
        }  # If requesting additional information
        if data_format == 'numpy':
            d = {
                'w': osapo_df.drop(['treatment', 'outcome', 'ite'], axis='columns').to_numpy(),
                't': osapo_df['treatment'].to_numpy(),
                'y': osapo_df['outcome'].to_numpy()
            }
            if return_ites:
                d['ite'] = osapo_df['ite'].to_numpy()
            if ret_counterfactual_outcomes:
                d['y0'] = osapo_df['counterfactual_outcome_0'].to_numpy()
                d['y1'] = osapo_df['counterfactual_outcome_1'].to_numpy()
        elif data_format == 'pandas':
            d = {
                'w': osapo_df.drop(['treatment', 'outcome', 'ite'], axis='columns'),
                't': osapo_df['treatment'],
                'y': osapo_df['outcome']
            }
            if return_ites:
                d['ite'] = osapo_df['ite']
            if ret_counterfactual_outcomes:
                d['y0'] = osapo_df['counterfactual_outcome_0']
                d['y1'] = osapo_df['counterfactual_outcome_1']
        else:
            raise ValueError(f"Data format {data_format} not supported.")
        return d
