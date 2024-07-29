"""Script to load the APO data (from previous versions of ACIC). This will then be
used to generate observational data with a specific biasing function. This data
will serve as the source dataset for Realcause."""

# Import libraries
import pandas as pd
import numpy as np
from consts import BASE_DATASETS_FOLDER
from utilities import biasing_function

def get_apo_data(identifier, data_format='numpy', return_ites=True, **kwargs):
    """Function to get an APO dataset that was used in the CausalEval paper.
    For now we only implement one specific ACIC dataset. 
    """
    if identifier == 'acic':
        df = pd.read_csv(f'{BASE_DATASETS_FOLDER}/apo_acic_4_data.csv', index_col=0)
        # Read in the config
        cfg = pd.read_csv(f'{BASE_DATASETS_FOLDER}/apo_acic_4_config.txt',
                          sep=' ',
                          index_col=None,
                          names=['column', 'type'])
        # If a column exists in the df.columns, but is not present in the cfg['column'], then drop it
        df = df[[
            x for x in df.columns.tolist() if x in [
                *cfg['column'].tolist(), 'counterfactual_outcome_1',
                'counterfactual_outcome_0'
            ]
        ]]
        # Except for the index column, extract all the other columns of type = 'f' in the cfg file
        categorical_var = cfg.loc[cfg['type'] == 'f', 'column'].tolist()
        # Remove the index column from the list
        categorical_var.remove('index')
        # All other columns are continuous variables
        continuous_vars = cfg.loc[cfg['type'] == 'n', 'column'].tolist()
        # If the following variables are present in categorical_var, then remove them
        one_hot_vars = [
            x for x in categorical_var if x not in [
                'treatment', 'outcome', 'counterfactual_outcome_1',
                'counterfactual_outcome_0'
            ]
        ]
        df = pd.get_dummies(df,
                            columns=one_hot_vars,
                            drop_first=True,
                            dtype=np.float64)
        # Get a list of the updated categorical variables
        updated_categorical_vars = [
            x for x in df.columns.tolist() if x not in continuous_vars
        ]
        # Remove counterfacutal outcomes from the continuous variables
        updated_categorical_vars = [
            x for x in updated_categorical_vars if x not in
            ['counterfactual_outcome_1', 'counterfactual_outcome_0']
        ]
        # Get the biasing covariate (assume single biasing covariate for now)
        biasing_covariate = cfg.iloc[3]['column']
        # Create an observational dataset from the APO dataset with the desired parameters
        intercept = kwargs['intercept'] if 'intercept' in kwargs.keys() else 0  # pylint: disable=consider-iterating-dictionary
        weight = kwargs['weight'] if 'weight' in kwargs.keys() else 1  # pylint: disable=consider-iterating-dictionary
        osapo_df = biasing_function.osrct_algorithm(
            df,
            confound_func_params={
                'para_form': 'linear',
                'intercept': intercept,
                'weight': weight
            },
            treatment_col='treatment',
            confounding_vars=[biasing_covariate])
        # Compute ITE, expected difference between counterfactual outcomes, after subsampling
        osapo_df['ites'] = osapo_df['counterfactual_outcome_1'] - osapo_df[
            'counterfactual_outcome_0']
        # Compute ATE, mean of the ITE after subsampling
        ate = osapo_df['ites'].mean()
        # Compute the naive ATE
        naive_ate = osapo_df.loc[osapo_df['treatment'] == 1, 'outcome'].mean(
        ) - osapo_df.loc[osapo_df['treatment'] == 0, 'outcome'].mean()
        # Return the dataset and the ATE, as well as the other required information
        df_info = {
            'post_treatment_vars': ['outcome'],
            'treatment_var': ['treatment'],
            'numerical_vars': continuous_vars,
            'categorical_vars': updated_categorical_vars,
            'true_ate': ate,
            'naive_ate': naive_ate,
            'biasing_vars': [biasing_covariate],
        }
        print(f'Information: {df_info}')
        # Drop the following columns from the dataframe
        osapo_df.drop(columns=[
            'counterfactual_outcome_1', 'counterfactual_outcome_0',
        ], inplace=True)

        if data_format == 'numpy':
            d = {
                'w': osapo_df.drop(['treatment', 'outcome', 'ites'], axis='columns').to_numpy(),
                't': osapo_df['treatment'].to_numpy(),
                'y': osapo_df['outcome'].to_numpy()
            }
            if return_ites: 
                d['ites'] = osapo_df['ites'].to_numpy()
        elif data_format == 'pandas':
            d = {
                'w': osapo_df.drop(['treatment', 'outcome', 'ites'], axis='columns'),
                't': osapo_df['treatment'],
                'y': osapo_df['outcome']
            }
            if return_ites: 
                d['ites'] = osapo_df['ites']
        else:
            raise ValueError(f"Data format {data_format} not supported.")
        return d
        