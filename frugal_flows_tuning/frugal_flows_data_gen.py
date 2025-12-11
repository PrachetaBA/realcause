"""Script to generate data from the FrugalFlows models
according to the tuned hyperparameters.

Conda environment: rc-ff-sbi
"""

# Import libraries
import os
import argparse
import logging
import pandas as pd
import yaml
from tqdm import tqdm

import jax
import jax.random as jr
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

# Import dataloader
from loading import load_gen
from data_loaders import lalonde as rc_lalonde    # Data loaders for Realcause simulator
from data_loaders import apo as rc_apo    # Data loaders for APO simulator
from frugal_flows.benchmarking import FrugalFlowModel

# Defing logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_ff_data(config, expt_id, num_samples=50):
    """Function to generate data from the FrugalFlows models
    according to the tuned hyperparameters."""
    # Extract the configuration parameters for the experiment
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
    elif dataset_name == 'postgres':
        rc_model_path = 'results/realcause_models/postgres_linear_3000/default'
        d, d_info = rc_apo.get_apo_data(identifier='postgres', confound_func=dataset_identifier, data_format='pandas', return_ites=False, ret_counterfactual_outcomes=False, sample_size=sample_size)
        outcome_col = d_info['outcome_col']
        treatment_col = d_info['treatment_col']
        categorical_vars = d_info['categorical_vars']    # Excludes T and Y
        continuous_vars = d_info['continuous_vars']    # Excludes T and Y
        covariates_col = continuous_vars + categorical_vars
        covariates_df = d[
            'w'].values    # This is the original covariates dataframe (not transformed)
        d = pd.concat([d['w'], d['t'], d['y']], axis=1)
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented')

    # Load the Realcause model from the specified path (before applying transformations)
    # We need to use the model's transforms to ensure scales match
    rc_model, _ = load_gen(saveroot=rc_model_path)

    # Apply transformations using the model's transforms (if specified in config)
    # This ensures the observed data is normalized using the same parameters as the model
    if dataset_name in ['lalonde', 'twins']:
        if config['transform'] == True:
            print(f'The covariates columns are: {covariates_col}')
            # The model's w_transform was created from training data
            # Transform the covariates using the model's transform
            transformed_w = rc_model.w_transform.transform(d[covariates_col].values)
            # Assign back to DataFrame columns
            for i, col in enumerate(covariates_col):
                d[col] = transformed_w[:, i]
            # Do the same for the outcome column
            transformed_y = rc_model.y_transform.transform(d[outcome_col].values.reshape(-1, 1))
            d[outcome_col] = transformed_y.flatten()
        # This column order is compatible with the FrugalFlows simulator as well
        d = d[[outcome_col, treatment_col] + covariates_col]

    # After doing the transformations, we need to find the true ATE using the lalonde rct data
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

    # Training the FrugalFlows model
    # Convert the columns of the observed data to JAX compatible arrays
    X = jnp.array(d[treatment_col].values, dtype=jnp.float64)[:, None]
    Y = jnp.array(d[outcome_col].values, dtype=jnp.float64)[:, None]
    if len(categorical_vars) > 0:
        Z_disc = jnp.array(d[categorical_vars].values, dtype=jnp.float64)
    else:
        Z_disc = None
    if len(continuous_vars) > 0:
        Z_cont = jnp.array(d[continuous_vars].values, dtype=jnp.float64)
    else:
        Z_cont = None

    # Read in the hyperparameters for the trained models
    with open(config['frugalflows_hp_file'], 'r', encoding='utf-8') as file:
        tuned_hyperparams = yaml.safe_load(file)

    # Set tuning parameters for Normalizing Flows
    max_patience = config.get('max_patience', 200)    # Default is 200 TODO: Change after testing
    max_epochs = config.get('max_epochs', 5000)    # Default is 5000 TODO: Change after testing
    tuned_hyperparams['hyperparameters']['max_patience'] = max_patience
    tuned_hyperparams['hyperparameters']['max_epochs'] = max_epochs

    # Train the frugal flow model with the hyperparameters specified in the config
    trained_ff_model = FrugalFlowModel(X=X,
                                       Y=Y,
                                       Z_disc=Z_disc,
                                       Z_cont=Z_cont,
                                       confounding_copula=None)
    # Set default causal model and corresponding arguments
    causal_model = config.get('causal_model', 'location_translation')
    causal_model_args = None
    if causal_model == 'location_translation':
        trained_ff_model.train_benchmark_model(
            training_seed=jr.PRNGKey(tuned_hyperparams['seed']),
            marginal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            frugal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            causal_model='location_translation',
            causal_model_args={
                'ate': tuned_hyperparams['ate'], **tuned_hyperparams['cm_hyperparameters']
            },
            prop_flow_hyperparam_dict=tuned_hyperparams['hyperparameters'],
        )
        # Print out the causal margin obtained after training the FF model
        learned_causal_margin = trained_ff_model.frugal_flow.bijection.bijections[-1].bijections[
            0].ate
        logger.info(f'Learned causal margin: {learned_causal_margin}')
    elif causal_model == 'gaussian':
        trained_ff_model.train_benchmark_model(
            training_seed=jr.PRNGKey(tuned_hyperparams['seed']),
            marginal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            frugal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            prop_flow_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            causal_model='gaussian',
            causal_model_args={
                'ate': jnp.array([0.]),    # Starting values
                'const': 0.,
                'scale': 1
            })
        # Print out the causal margin obtained after training the FF model
        learned_causal_margin = trained_ff_model.frugal_flow.bijection.bijections[
            -1].bijection.bijections[0]
        logger.info(f'Learned causal margin: {learned_causal_margin.ate[0]}')
        logger.info(f'Learned causal margin: {learned_causal_margin.const}')
        logger.info(f'Learned scale: {learned_causal_margin.scale}')
        causal_model_args = {
            'const': learned_causal_margin.const, 'scale': learned_causal_margin.scale
        }

    rho = config.get('rho', 0.0)
    ate_setting = config.get('ate', 'flexible_ate')
    if ate_setting == 'flexible_ate':
        ate = learned_causal_margin
    elif ate_setting == 'true_ate':
        ate = true_ate
    elif ate_setting == 'incorrect_ate':
        ate = config.get('ate_value', 10.0)
    else:
        raise ValueError(f'Treatment effect {ate_setting} not implemented')

    # Create the directory to store the generated data
    generated_data_dir = f'data/generated_datasets/frugalflows/expt_{expt_id}'
    os.makedirs(generated_data_dir, exist_ok=True)

    # Generate the data
    for itr in tqdm(range(num_samples)):
        if causal_model == 'location_translation':
            generated_df = trained_ff_model.generate_samples(
                key=jr.PRNGKey(10 * 1),
                sampling_size=d.shape[0],    # This is the size of the original Lalonde dataset
                copula_param=rho,    # This is the correlation parameter
                outcome_causal_model='location_translation',
                outcome_causal_args={'ate': ate},
                with_confounding=True)
        elif causal_model == 'gaussian':
            generated_df = trained_ff_model.generate_samples(
                key=jr.PRNGKey(10 * 1),
                sampling_size=d.shape[0],    # This is the size of the original Lalonde dataset
                copula_param=rho,    # This is the correlation parameter
                outcome_causal_model='causal_cdf',
                outcome_causal_args={
                    'ate': jnp.array([ate]),
                    'const': causal_model_args['const'],
                    'scale': causal_model_args['scale']
                },
                with_confounding=True)
        # Depending on the dataset identifier, we may have to return specific columns
        if dataset_name == 'lalonde':
            generated_df.columns = [
                're78',
                'treat',
                'age',
                'education',
                're75',
                're74',
                'black',
                'hispanic',
                'married',
                'nodegree'
            ]
        elif dataset_name == 'postgres':
            generated_df.columns = [outcome_col, treatment_col, *covariates_col]
        # Save the generated dataset
        generated_df.to_csv(f'{generated_data_dir}/dataset_{itr}.csv', index=False)


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_identifier', type=str, required=True)
    args = parser.parse_args()

    expt_id = args.experiment_identifier
    # Load the configuration file
    with open(args.config_file, 'r', encoding='utf-8') as file:
        all_experiment_configs = yaml.safe_load(file)
    config = all_experiment_configs[f'expt_{expt_id}']

    # Generate the data according to the specific experiment
    generate_ff_data(config, expt_id, num_samples=50)
