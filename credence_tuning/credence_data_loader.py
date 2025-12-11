"""Data loader specific to the Credence models, we need to also load the corresponding
realcause model so that we can normalize the data in the same way."""

# Import libraries
from loading import load_gen
import pandas as pd
from data_loaders import lalonde as rc_lalonde
from data_loaders import apo as rc_apo


def load_data_credence(dataset_name, dataset_identifier=None, sample_size=None, rc_model_path=None):
    """Function to load the data defined by the identifier
    specifically for use for Credence models."""
    dataset_info = {}
    if dataset_name == 'lalonde':
        if dataset_identifier == 'psid1':
            d = rc_lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
            rc_model_path = 'results/GenModelCkpts/lalonde/psid1/save'
        elif dataset_identifier == 'cps1':
            d = rc_lalonde.load_lalonde(obs_version='cps', data_format='pandas_single')
            rc_model_path = 'results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize'
        else:
            raise ValueError(f'Dataset identifier {dataset_identifier} not implemented')
        d.drop(columns=['data_id'], inplace=True)
        outcome_col = 're78'
        treatment_col = 'treat'
        categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
        continuous_vars = ['age', 'education', 're75', 're74']
        covariates_col = continuous_vars + categorical_vars

        # Apply the transformations to the data as done in the Realcause model
        rc_model, _ = load_gen(rc_model_path)
        transformed_w = rc_model.w_transform.transform(d[covariates_col].values)
        # Assign back to DataFrame columns
        for i, col in enumerate(covariates_col):
            d[col] = transformed_w[:, i]
        # Do the same for the outcome column
        transformed_y = rc_model.y_transform.transform(d[outcome_col].values.reshape(-1, 1))
        d[outcome_col] = transformed_y.flatten()

        # Find the true ATE using the RCT data in the transformed space
        lalonde_rct = rc_lalonde.load_lalonde(rct=True, data_format='pandas_single')
        lalonde_rct.drop(columns=['data_id'], inplace=True)
        # Transform the RCT data using the same transformations
        transformed_w_rct = rc_model.w_transform.transform(lalonde_rct[covariates_col].values)
        for i, col in enumerate(covariates_col):
            lalonde_rct[col] = transformed_w_rct[:, i]
        # Do the same for the outcome column
        transformed_y_rct = rc_model.y_transform.transform(lalonde_rct[outcome_col].values.reshape(
            -1, 1))
        lalonde_rct[outcome_col] = transformed_y_rct.flatten()
        true_ate = lalonde_rct[outcome_col][lalonde_rct['treat'] == 1].mean(
        ) - lalonde_rct[outcome_col][lalonde_rct['treat'] == 0].mean()

        # Store the dataset information
        dataset_info = {
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'covariates_col': covariates_col,
            'categorical_vars': categorical_vars +
                                [treatment_col
                                ],    # Include the treatment column in the categorical variables
            'continuous_vars': continuous_vars +
                               [outcome_col
                               ],    # Include the outcome column in the continuous variables
            'true_ate': true_ate
        }
    elif dataset_name == 'postgres':
        d, df_info = rc_apo.get_apo_data(identifier='postgres', confound_func=dataset_identifier, data_format='pandas', return_ites=False, ret_counterfactual_outcomes=False, sample_size=sample_size)
        d = pd.concat([d['w'], d['t'], d['y']], axis=1)
        outcome_col = df_info['outcome_col']
        treatment_col = df_info['treatment_col']
        covariates_col = df_info['categorical_vars'] + df_info['continuous_vars']
        updated_categorical_vars = df_info['categorical_vars'] + [
            treatment_col
        ]    # Include the treatment column in the categorical variables
        updated_continuous_vars = df_info['continuous_vars'] + [
            outcome_col
        ]    # Include the outcome column in the continuous variables
        true_ate = df_info['true_ate']
        dataset_info = {
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'covariates_col': covariates_col,
            'categorical_vars': updated_categorical_vars,
            'continuous_vars': updated_continuous_vars,
            'true_ate': true_ate
        }

    return d, dataset_info


# Test the data_loader for credence for the postgres dataset
if __name__ == '__main__':
    d, dataset_info = load_data_credence(dataset_name='postgres', dataset_identifier='linear', sample_size=3000, rc_model_path='results/realcause_models/postgres_linear_3000/default')
    print(d.head())
    print(dataset_info)
