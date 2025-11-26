# pylint: disable=redefined-outer-name
"""Script to generate data from the Realcause models
according to the tuned models and specific DGP parameters.

Conda environment: rc-ff-sbi"""

# Import libraries
import os
import logging
import yaml
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
from loading import load_gen
from data_loaders import lalonde as rc_lalonde

# Defing logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_rc_data(config, expt_id, num_samples=50):
    """Function to generate data from the Realcause models
    according to the tuned models and specific DGP parameters."""
    dataset_name = config['dataset_name']
    dataset_identifier = config['dataset_identifier']
    sample_size = config['sample_size']

    # Load the observed dataset to be used as the reference dataset
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
        # Sort the covariates columns to put the continous first, then the categorical
        covariates_col = continuous_vars + categorical_vars
        covariates_df = d[
            covariates_col].values    # This is the original covariates dataframe (not transformed)

    # Load the Realcause model from the specified path (before applying transformations)
    # We need to use the model's transforms to ensure scales match
    rc_model, _ = load_gen(saveroot=rc_model_path)

    # Extract the true ATE from the RCT data
    if dataset_name == 'lalonde':
        # Find the true ATE using the RCT data in the transformed space
        lalonde_rct = rc_lalonde.load_lalonde(rct=True, data_format='pandas_single')
        lalonde_rct.drop(columns=['data_id'], inplace=True)
        # Transform the RCT data using the same transformations
        transformed_w_rct = rc_model.w_transform.transform(lalonde_rct[covariates_col].values)
        for i, col in enumerate[str](covariates_col):
            lalonde_rct[col] = transformed_w_rct[:, i]
        # Do the same for the outcome column
        transformed_y_rct = rc_model.y_transform.transform(lalonde_rct[outcome_col].values.reshape(
            -1, 1))
        lalonde_rct[outcome_col] = transformed_y_rct.flatten()
        true_ate = lalonde_rct[outcome_col][lalonde_rct['treat'] == 1].mean(
        ) - lalonde_rct[outcome_col][lalonde_rct['treat'] == 0].mean()

    overlap = 1.0
    deg_hetero = 1.0
    if config.get('transform', True):
        untransform = False
    else:
        untransform = True

    ate_setting = config.get('ate_setting', 'flexible_ate')
    if ate_setting == 'flexible_ate':
        ate = None
    elif ate_setting == 'true_ate':
        ate = true_ate
    elif ate_setting == 'incorrect_ate':
        ate = config.get('ate_value', 10.0)
    else:
        raise ValueError(f'Treatment effect {ate_setting} not implemented')
    overlap = config.get('overlap', 1.0)
    deg_hetero = config.get('deg_hetero', 1.0)

    # Create the directory to store the generated data
    generated_data_dir = f'data/generated_datasets/realcause/expt_{expt_id}'
    os.makedirs(generated_data_dir, exist_ok=True)

    for itr in tqdm(range(num_samples)):    # TEMP: range(1)
        w, t, y = rc_model.sample(covariates_df,
                                causal_effect_scale=ate,
                                overlap=overlap,
                                deg_hetero=deg_hetero,
                                ret_counterfactuals=False,
                                untransform=untransform)
        # Ensure t and y are column vectors
        t = t.reshape(-1, 1) if t.ndim == 1 else t
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        # Concatenate arrays horizontally
        generated_data = np.column_stack([y, t, w])
        generated_df = pd.DataFrame(generated_data,
                                    columns=[outcome_col, treatment_col] + covariates_col)
        # Save the generated dataset
        generated_df.to_csv(f'{generated_data_dir}/dataset_{itr}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate datasets given the best model path.')
    parser.add_argument('--config_file',
                        type=str,
                        help='Path to the configuration file.',
                        default=None)
    parser.add_argument('--experiment_identifier',
                        type=str,
                        help='Identifier for the experiment.',
                        default=None)
    args = parser.parse_args()
    expt_id = args.experiment_identifier
    # Load the configuration file
    with open(args.config_file, 'r', encoding='utf-8') as file:
        all_experiment_configs = yaml.safe_load(file)
    config = all_experiment_configs[f'expt_{expt_id}']

    # Generate the datasets
    generate_rc_data(config, expt_id)
