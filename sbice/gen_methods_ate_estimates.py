"""Script to generate the ATE estimates for the different settings and
the different generative methods.

Conda environment: /work/pi_jensen_umass_edu/pboddavarama_umass_edu/pba-conda/envs/rpy
"""

# Import libraries
import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm
import yaml
from . import ate_estimators
from loading import load_gen
from data_loaders import lalonde, apo

# Set logger to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants that are used in this script
ALL_ESTIMATORS = [
    'Gradient Boosting Trees DML',
    'Linear DML',
    'Doubly Robust',
    'Linear T Learner',
    'Linear S Learner',
    'Linear X Learner',
    'Gradient Boosting Trees T Learner',
    'Gradient Boosting Trees S Learner',
    'Gradient Boosting Trees X Learner',
    'Random Forest T Learner',
    'Random Forest S Learner',
    'Random Forest X Learner',
    'Causal BART',
    'Causal Forest',
    'TMLE'
]
META_ESTIMATORS = [
    'Linear T Learner',
    'Linear S Learner',
    'Linear X Learner',
    'Random Forest T Learner',
    'Random Forest S Learner',
    'Random Forest X Learner',
    'Causal Forest'
]
# Define the path to save the estimated ATEs
ESTIMATED_ATE_PATH = 'output/gen_methods_ate_estimates'
NUM_SAMPLES = 50


def source_data_ate(dataset_name,
                    dataset_identifier=None,
                    sample_size=None,
                    experiment_identifier=None,
                    set_of_estimators=ALL_ESTIMATORS,
                    ate_df_folder=None,
                    categorical_conversion=True):
    """Function to load the source dataset and compute the ATE
    estimators and the true ATE of the dataset."""
    # Load the observed dataset to be used as the reference dataset
    if dataset_name == 'lalonde':
        if dataset_identifier == 'psid1':
            d = lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
            realcause_model_path = 'results/GenModelCkpts/lalonde/psid1/save'
        elif dataset_identifier == 'cps1':
            d = lalonde.load_lalonde(obs_version='cps', data_format='pandas_single')
            realcause_model_path = 'results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize'
        else:
            raise ValueError(f'Dataset identifier {dataset_identifier} not implemented')
        d.drop(columns=['data_id'], inplace=True)
        outcome_col = 're78'
        treatment_col = 'treat'
        covariates_col = d.columns.tolist()
        covariates_col.remove(outcome_col)
        covariates_col.remove(treatment_col)

        # Load the Realcause model
        rc_model, _ = load_gen(saveroot=realcause_model_path)

        # We want to apply the ATE estimators to the transformed data
        # Transform the covariates using the model's transform
        transformed_w = rc_model.w_transform.transform(d[covariates_col].values)
        # Assign back to DataFrame columns
        for i, col in enumerate(covariates_col):
            d[col] = transformed_w[:, i]
        # Do the same for the outcome column
        transformed_y = rc_model.y_transform.transform(d[outcome_col].values.reshape(-1, 1))
        d[outcome_col] = transformed_y.flatten()
        # Reorder the columns to put all the covariates first, then treatment, then outcome
        d = d[covariates_col + [treatment_col, outcome_col]]

        # Apply the transformations to the RCT data to compute the true ATE
        rct_data = lalonde.load_lalonde(rct_version='dw', rct=True, data_format='pandas_single')
        rct_data[covariates_col] = rc_model.w_transform.transform(rct_data[covariates_col].values)
        rct_data[outcome_col] = rc_model.y_transform.transform(rct_data[outcome_col].values.reshape(
            -1, 1))
        true_ate = rct_data['re78'][rct_data['treat'] == 1].mean() - rct_data['re78'][
            rct_data['treat'] == 0].mean()

        # We want to transform the categorical variables back to 1 and 0s if applicable
        if categorical_conversion:
            # Transform the following covariates back to 1 and 0s
            categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
            for col in categorical_vars:
                # If value = 0, 0 otherwise 1
                d[col] = d[col].apply(lambda x: 0.0 if x == 0 else 1.0)

    elif dataset_name == 'postgres':
        d, d_info = apo.get_apo_data(identifier='postgres', confound_func=dataset_identifier, data_format='pandas', return_ites=False, ret_counterfactual_outcomes=False, sample_size=sample_size)
        true_ate = d_info['true_ate']
        treatment_col = d_info['treatment_col']
        outcome_col = d_info['outcome_col']
        # Concatenate the covariates, treatment and outcome columns
        d = pd.concat([d['w'], d['t'], d['y']], axis=1)
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented')

    # Run the ATE estimators for the source dataset
    estimated_ate = ate_estimators.bootstrap_ate_inference(
        outcome=outcome_col,
        treatment=treatment_col,
        data=d,
        dataset_identifier=f'{dataset_name}_{dataset_identifier}_{sample_size}',
        set_of_estimators=set_of_estimators,
        repeats=1)
    # Save the estimated ATEs to a csv file
    estimated_ate['df'] = f'{dataset_name}_{dataset_identifier}_{sample_size}_source'
    estimated_ate['true_ate'] = true_ate

    ATE_DF_FILENAME = f'{ate_df_folder}{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_source_ate.csv'
    estimated_ate.to_csv(ATE_DF_FILENAME, index=False)
    logger.info(f'Estimated ATEs for the source dataset saved to {ATE_DF_FILENAME}')


def credence_data_ate(config_file,
                      experiment_identifier,
                      gen_data_dir,
                      set_of_estimators=ALL_ESTIMATORS,
                      ate_df_folder=None):
    """Function to load the datasets generated by the Credence model
    for the different settings and compute the ATE estimators."""
    with open(config_file, 'r') as f:
        all_expt_configs = yaml.safe_load(f)
    config = all_expt_configs[f'expt_{experiment_identifier}']
    dataset_name = config['dataset_name']
    dataset_identifier = config['dataset_identifier']
    sample_size = config['sample_size']

    # Map the outcome and treatment columns for the different datasets
    if dataset_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif dataset_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'

    # Load the datasets generated by the Credence model
    ate_df = pd.DataFrame()
    for itr in tqdm(range(NUM_SAMPLES)):
        df_gen = pd.read_csv(f'{gen_data_dir}/dataset_{itr}.csv')
        # We want to create the outcome column
        df_gen[outcome_col] = (df_gen[treatment_col] * df_gen['Y1']) + (
            (1 - df_gen[treatment_col]) * df_gen['Y0'])
        # Compute the true ATE for the generated dataset
        df_gen['ite'] = df_gen['Y1'] - df_gen['Y0']
        true_ate = df_gen['ite'].mean()
        # Drop the unnecessary columns
        df_gen.drop(columns=['Y1', 'Y0', 'ite'], inplace=True)

        # Run the ATE estimators for the generated dataset
        estimated_ate = ate_estimators.bootstrap_ate_inference(
            outcome=outcome_col,
            treatment=treatment_col,
            data=df_gen,
            dataset_identifier=
            f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_credence_{itr}',
            set_of_estimators=set_of_estimators,
            repeats=1)
        # Save the estimated ATEs to a csv file
        estimated_ate[
            'df'] = f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_credence_{itr}'
        estimated_ate['true_ate'] = true_ate
        ate_df = pd.concat([ate_df, estimated_ate])

    ATE_DF_FILENAME = f'{ate_df_folder}{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_credence_ate.csv'
    ate_df.to_csv(ATE_DF_FILENAME, index=False)
    logger.info(f'Estimated ATEs for the generated dataset saved to {ATE_DF_FILENAME}')


def mcredence_data_ate(config_file,
                       experiment_identifier,
                       gen_data_dir,
                       set_of_estimators=ALL_ESTIMATORS,
                       ate_df_folder=None):
    """Function to load the datasets generated by the Modified Credence model
    for the different settings and compute the ATE estimators."""
    with open(config_file, 'r') as f:
        all_expt_configs = yaml.safe_load(f)
    config = all_expt_configs[f'expt_{experiment_identifier}']
    dataset_name = config['dataset_name']
    dataset_identifier = config['dataset_identifier']
    sample_size = config['sample_size']

    # Map the outcome and treatment columns for the different datasets
    if dataset_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif dataset_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'

    # Load the datasets generated by the Modified Credence model
    ate_df = pd.DataFrame()
    for itr in tqdm(range(NUM_SAMPLES)):
        df_gen = pd.read_csv(f'{gen_data_dir}/dataset_{itr}.csv')
        # We want to create the outcome column
        df_gen[outcome_col] = (df_gen[treatment_col] * df_gen['Y1']) + (
            (1 - df_gen[treatment_col]) * df_gen['Y0'])
        # Compute the true ATE for the generated dataset
        df_gen['ite'] = df_gen['Y1'] - df_gen['Y0']
        true_ate = df_gen['ite'].mean()
        # Drop the unnecessary columns
        df_gen.drop(columns=['Y1', 'Y0', 'ite'], inplace=True)

        # Run the ATE estimators for the generated dataset
        estimated_ate = ate_estimators.bootstrap_ate_inference(
            outcome=outcome_col,
            treatment=treatment_col,
            data=df_gen,
            dataset_identifier=
            f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_mcredence_{itr}',
            set_of_estimators=set_of_estimators,
            repeats=1)
        # Save the estimated ATEs to a csv file
        estimated_ate[
            'df'] = f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_mcredence_{itr}'
        estimated_ate['true_ate'] = true_ate
        ate_df = pd.concat([ate_df, estimated_ate])

    ATE_DF_FILENAME = f'{ate_df_folder}{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_mcredence_ate.csv'
    ate_df.to_csv(ATE_DF_FILENAME, index=False)
    logger.info(f'Estimated ATEs for the generated dataset saved to {ATE_DF_FILENAME}')


def frugalflows_data_ate(config_file,
                         experiment_identifier,
                         gen_data_dir,
                         set_of_estimators=ALL_ESTIMATORS,
                         ate_df_folder=None,
                         categorical_conversion=True):
    """Function to load the datasets generated by the Frugal Flows model
    for the different settings and compute the ATE estimators."""
    with open(config_file, 'r') as f:
        all_expt_configs = yaml.safe_load(f)
    config = all_expt_configs[f'expt_{experiment_identifier}']
    dataset_name = config['dataset_name']
    dataset_identifier = config['dataset_identifier']
    sample_size = config['sample_size']

    # Map the outcome and treatment columns for the different datasets
    if dataset_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif dataset_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'

    # Manually input the learned causal margin as the true ATE depending on the experiment identifier
    if experiment_identifier == '0001':
        true_ate = 4.848423146302874
    elif experiment_identifier == '0002':
        true_ate = 0.016190112
    elif experiment_identifier == '0003':
        true_ate = 10.0
    elif experiment_identifier == '0004':
        true_ate = -0.494372492680495
    else:
        raise ValueError(f'Experiment identifier {experiment_identifier} not implemented')

    # Load the datasets generated by the Frugal Flows model
    ate_df = pd.DataFrame()
    for itr in tqdm(range(NUM_SAMPLES)):
        df_gen = pd.read_csv(f'{gen_data_dir}/dataset_{itr}.csv')
        if categorical_conversion:
            if dataset_name == 'lalonde':
                # Transform the following covariates back to 1 and 0s
                categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
                for col in categorical_vars:
                    # If value = 0, 0 otherwise 1
                    df_gen[col] = df_gen[col].apply(lambda x: 0.0 if x == 0 else 1.0)
        # Run the ATE estimators for the generated dataset
        estimated_ate = ate_estimators.bootstrap_ate_inference(
            outcome=outcome_col,
            treatment=treatment_col,
            data=df_gen,
            dataset_identifier=
            f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_frugalflows_{itr}',
            set_of_estimators=set_of_estimators,
            repeats=1)
        # Save the estimated ATEs to a csv file
        estimated_ate[
            'df'] = f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_frugalflows_{itr}'
        estimated_ate['true_ate'] = true_ate
        ate_df = pd.concat([ate_df, estimated_ate])

    ATE_DF_FILENAME = f'{ate_df_folder}{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_frugalflows_ate.csv'
    ate_df.to_csv(ATE_DF_FILENAME, index=False)
    logger.info(f'Estimated ATEs for the generated dataset saved to {ATE_DF_FILENAME}')


def realcause_data_ate(config_file,
                       experiment_identifier,
                       gen_data_dir,
                       set_of_estimators=ALL_ESTIMATORS,
                       ate_df_folder=None,
                       categorical_conversion=True):
    """Function to load the datasets generated by the Realcause model
    for the different settings and compute the ATE estimators."""
    with open(config_file, 'r') as f:
        all_expt_configs = yaml.safe_load(f)
    config = all_expt_configs[f'expt_{experiment_identifier}']
    dataset_name = config['dataset_name']
    dataset_identifier = config['dataset_identifier']
    sample_size = config['sample_size']

    # Map the outcome and treatment columns for the different datasets
    if dataset_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif dataset_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented')

    # Load the datasets generated by the Realcause model
    ate_df = pd.DataFrame()
    for itr in tqdm(range(NUM_SAMPLES)):
        df_gen = pd.read_csv(f'{gen_data_dir}/dataset_{itr}.csv')
        # Compute the true ATE for the generated dataset
        df_gen['ite'] = df_gen['y1'] - df_gen['y0']
        true_ate = df_gen['ite'].mean()
        # Drop the unnecessary columns
        df_gen.drop(columns=['y1', 'y0', 'ite'], inplace=True)
        if categorical_conversion:
            if dataset_name == 'lalonde':
                # Transform the following covariates back to 1 and 0s
                categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
                for col in categorical_vars:
                    # If value = 0, 0 otherwise 1
                    df_gen[col] = df_gen[col].apply(lambda x: 0.0 if x == 0 else 1.0)
        # Run the ATE estimators for the generated dataset
        estimated_ate = ate_estimators.bootstrap_ate_inference(
            outcome=outcome_col,
            treatment=treatment_col,
            data=df_gen,
            dataset_identifier=
            f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_realcause_{itr}',
            set_of_estimators=set_of_estimators,
            repeats=1)
        # Save the estimated ATEs to a csv file
        estimated_ate[
            'df'] = f'{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_realcause_{itr}'
        estimated_ate['true_ate'] = true_ate
        ate_df = pd.concat([ate_df, estimated_ate])

    ATE_DF_FILENAME = f'{ate_df_folder}{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_realcause_ate.csv'
    ate_df.to_csv(ATE_DF_FILENAME, index=False)
    logger.info(f'Estimated ATEs for the generated dataset saved to {ATE_DF_FILENAME}')


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_identifier', type=str, required=True)
    parser.add_argument('--sample_size', type=str, default=None, required=False)
    parser.add_argument('--experiment_identifier', type=str, required=True)
    parser.add_argument('--gen_method', type=str, default='source', required=True)
    args = parser.parse_args()

    ate_df_folder = f'{ESTIMATED_ATE_PATH}/{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}/'
    if not os.path.exists(ate_df_folder):
        os.makedirs(ate_df_folder)
    else:
        logger.info(f'Exists already')

    if args.gen_method == 'source':
        source_data_ate(args.dataset_name,
                        args.dataset_identifier,
                        args.sample_size,
                        args.experiment_identifier,
                        set_of_estimators=ALL_ESTIMATORS,
                        ate_df_folder=ate_df_folder,
                        categorical_conversion=True)
    elif args.gen_method == 'credence':
        credence_config_file = f'configs/credence_experiments.yaml'
        credence_gen_data_dir = f'data/generated_datasets/credence/expt_{args.experiment_identifier}'
        credence_data_ate(credence_config_file,
                          args.experiment_identifier,
                          credence_gen_data_dir,
                          set_of_estimators=ALL_ESTIMATORS,
                          ate_df_folder=ate_df_folder)
    elif args.gen_method == 'mcredence':
        mcredence_config_file = f'configs/credence_experiments.yaml'
        mcredence_gen_data_dir = f'data/generated_datasets/modified_credence/expt_{args.experiment_identifier}'
        mcredence_data_ate(mcredence_config_file,
                           args.experiment_identifier,
                           mcredence_gen_data_dir,
                           set_of_estimators=ALL_ESTIMATORS,
                           ate_df_folder=ate_df_folder)
    elif args.gen_method == 'frugalflows':
        frugalflows_config_file = f'configs/frugalflows_experiments.yaml'
        frugalflows_gen_data_dir = f'data/generated_datasets/frugalflows/expt_{args.experiment_identifier}'
        frugalflows_data_ate(frugalflows_config_file,
                             args.experiment_identifier,
                             frugalflows_gen_data_dir,
                             set_of_estimators=ALL_ESTIMATORS,
                             ate_df_folder=ate_df_folder,
                             categorical_conversion=True)
    elif args.gen_method == 'realcause':
        realcause_config_file = f'configs/realcause_experiments.yaml'
        realcause_gen_data_dir = f'data/generated_datasets/realcause/expt_{args.experiment_identifier}'
        realcause_data_ate(realcause_config_file,
                           args.experiment_identifier,
                           realcause_gen_data_dir,
                           set_of_estimators=ALL_ESTIMATORS,
                           ate_df_folder=ate_df_folder,
                           categorical_conversion=True)
    else:
        raise ValueError(f'Generation method {args.gen_method} not implemented')
