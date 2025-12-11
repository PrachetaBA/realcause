"""Script to compute the metrics for the datasets generated using different generative methods. We implement the following metrics:
1. Classifier AUC
2. Sliced-Wasserstein distance
3. MMD distance (potentially fast calculation)
"""

# Import libraries
import argparse
import os
import sys
import logging
import warnings
import yaml
from tqdm import tqdm

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from loading import load_gen
from data_loaders import apo, lalonde, twins
import torch
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy

# create default evaluator for doctests


def eval_step(engine, batch):
    return batch


metric = MaximumMeanDiscrepancy()
default_evaluator = Engine(eval_step)
metric.attach(default_evaluator, 'mmd')

# Set logger to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
METRICS_PATH = 'output/gen_methods_metrics'
NUM_SAMPLES = 50

##############################################################
# Functions to load the generated datasets #################################
##############################################################


def source_data_loader(dataset_name,
                       dataset_identifier=None,
                       sample_size=None,
                       experiment_identifier=None,
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
        # Let us reorder the columns to put all the covariates first, then treatment, then outcome
        d = d[[
            'rows',
            'creation_year',
            'num_ref_tables',
            'num_joins',
            'num_group_by',
            'queries_by_user',
            'length_chars',
            'total_ref_rows',
            treatment_col,
            outcome_col
        ]]
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented')

    return d, true_ate


def credence_data_loader(config_file, experiment_identifier, gen_data_dir):
    """Function to load the datasets generated by the Credence model
    for the different settings."""
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
    gen_datasets = []
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
        if dataset_name == 'postgres':
            # Let us reorder the columns to put all the covariates first, then treatment, then outcome
            df_gen = df_gen[[
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                treatment_col,
                outcome_col
            ]]
        gen_datasets.append(df_gen)
    return gen_datasets, true_ate


def mcredence_data_loader(config_file, experiment_identifier, gen_data_dir):
    """Function to load the datasets generated by the Modified Credence model
    for the different settings."""
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
    gen_datasets = []
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
        if dataset_name == 'postgres':
            # Let us reorder the columns to put all the covariates first, then treatment, then outcome
            df_gen = df_gen[[
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                treatment_col,
                outcome_col
            ]]
        gen_datasets.append(df_gen)
    return gen_datasets, true_ate


def frugalflows_data_loader(config_file,
                            experiment_identifier,
                            gen_data_dir,
                            categorical_conversion=True):
    """Function to load the datasets generated by the Frugal Flows model
    for the different settings."""
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
    gen_datasets = []
    for itr in tqdm(range(NUM_SAMPLES)):
        df_gen = pd.read_csv(f'{gen_data_dir}/dataset_{itr}.csv')
        if categorical_conversion:
            if dataset_name == 'lalonde':
                # Transform the following covariates back to 1 and 0s
                categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
                for col in categorical_vars:
                    # If value = 0, 0 otherwise 1
                    df_gen[col] = df_gen[col].apply(lambda x: 0.0 if x == 0 else 1.0)
        if dataset_name == 'postgres':
            # Let us reorder the columns to put all the covariates first, then treatment, then outcome
            df_gen = df_gen[[
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                treatment_col,
                outcome_col
            ]]
        gen_datasets.append(df_gen)
    return gen_datasets, true_ate


def realcause_data_loader(config_file,
                          experiment_identifier,
                          gen_data_dir,
                          categorical_conversion=True):
    """Function to load the datasets generated by the Realcause model
    for the different settings."""
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
    gen_datasets = []
    for itr in tqdm(range(NUM_SAMPLES)):
        df_gen = pd.read_csv(f'{gen_data_dir}/dataset_{itr}.csv')
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
        if dataset_name == 'postgres':
            # Let us reorder the columns to put all the covariates first, then treatment, then outcome
            df_gen = df_gen[[
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                treatment_col,
                outcome_col
            ]]
        gen_datasets.append(df_gen)
    return gen_datasets, true_ate


##############################################################
# Functions to compute the metrics ###########################
##############################################################
def compute_slicedwass_distance(source_array, gen_array):
    """Compute the sliced-Wasserstein distance between the source and the generated datasets."""
    # Convert to numpy arrays if they are DataFrames
    if isinstance(source_array, pd.DataFrame):
        source_array = source_array.values
    if isinstance(gen_array, pd.DataFrame):
        gen_array = gen_array.values

    # Ensure arrays are numpy arrays with float type
    source_array = np.asarray(source_array, dtype=np.float64)
    gen_array = np.asarray(gen_array, dtype=np.float64)

    # Compute the sliced-Wasserstein distance
    slicedwass_distance = ot.sliced.sliced_wasserstein_distance(source_array,
                                                                gen_array,
                                                                n_projections=50,
                                                                p=2)
    return slicedwass_distance


def compute_mmd_distance(source_array, gen_array):
    """Compute the MMD distance between the source and the generated datasets."""
    # Convert to numpy arrays if they are DataFrames
    if isinstance(source_array, pd.DataFrame):
        source_array = source_array.values
    if isinstance(gen_array, pd.DataFrame):
        gen_array = gen_array.values
    # Ensure arrays are numpy arrays with float type
    source_array = np.asarray(source_array, dtype=np.float64)
    gen_array = np.asarray(gen_array, dtype=np.float64)

    # Use pytorch ignite library to compute the MMD distance
    source_tensor = torch.tensor(source_array)
    gen_tensor = torch.tensor(gen_array)
    dist = default_evaluator.run([[source_tensor, gen_tensor]])
    return dist.metrics['mmd']


def compute_metrics(experiment_identifier, gen_method, metric='auc'):
    """Compute the metrics for the given dataset."""
    if gen_method == 'realcause':
        config_file = f'configs/realcause_experiments.yaml'
        gen_data_dir = f'data/generated_datasets/realcause/expt_{experiment_identifier}'
        gen_datasets, true_ate = realcause_data_loader(config_file, experiment_identifier, gen_data_dir)
    elif gen_method == 'credence':
        config_file = f'configs/credence_experiments.yaml'
        gen_data_dir = f'data/generated_datasets/credence/expt_{experiment_identifier}'
        gen_datasets, true_ate = credence_data_loader(config_file, experiment_identifier, gen_data_dir)
    elif gen_method == 'mcredence':
        config_file = f'configs/credence_experiments.yaml'
        gen_data_dir = f'data/generated_datasets/modified_credence/expt_{experiment_identifier}'
        gen_datasets, true_ate = mcredence_data_loader(config_file, experiment_identifier, gen_data_dir)
    elif gen_method == 'frugalflows':
        config_file = f'configs/frugalflows_experiments.yaml'
        gen_data_dir = f'data/generated_datasets/frugalflows/expt_{experiment_identifier}'
        gen_datasets, true_ate = frugalflows_data_loader(config_file, experiment_identifier, gen_data_dir)
    else:
        raise ValueError(f'Generation method {gen_method} not implemented')

    # Load the source dataset
    if experiment_identifier in ['0001', '0002', '0003']:
        dataset_name = 'lalonde'
        dataset_identifier = 'psid1'
        sample_size = None
        realcause_model_path = 'results/GenModelCkpts/lalonde/psid1/save'
        source_df, true_ate = source_data_loader(dataset_name, dataset_identifier, sample_size, realcause_model_path)
    elif experiment_identifier == '0004':
        dataset_name = 'postgres'
        dataset_identifier = 'linear'
        sample_size = 3000
        source_df, true_ate = source_data_loader(dataset_name, dataset_identifier, sample_size)
    else:
        raise ValueError(f'Experiment identifier {experiment_identifier} not implemented')

    if metric == 'auc':
        # Compute the classifier AUC for the source dataset with each of the generated datasets
        aucs = []
        for gen_dataset in gen_datasets:
            # Label the source and generated datasets
            gen_dataset['label'] = 0
            source_df['label'] = 1
            # Concatenate the source and generated datasets
            data = pd.concat([source_df, gen_dataset], axis=0)
            # Split the data into features and labels
            X = data.drop(columns=['label'])
            y = data['label']
            # Create a random forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(clf, X, y, scoring='roc_auc', n_jobs=-1)
            mean_auc = np.mean(scores)
            aucs.append(mean_auc)

        # Save the AUCs to a csv file
        aucs_df = pd.DataFrame(aucs, columns=['auc'])
        aucs_df.to_csv(f'{METRICS_PATH}/aucs_{experiment_identifier}_{gen_method}.csv', index=False)
    elif metric == 'slicedwass':
        # Compute the sliced wasserstein distances between the source and the generated datasets
        slicedwass_distances = []
        for gen_dataset in gen_datasets:
            slicedwass_distance = compute_slicedwass_distance(source_df, gen_dataset)
            slicedwass_distances.append(slicedwass_distance)

        # Save the sliced-Wasserstein distances to a csv file
        slicedwass_distances_df = pd.DataFrame(slicedwass_distances,
                                               columns=['slicedwass_distance'])
        slicedwass_distances_df.to_csv(
            f'{METRICS_PATH}/slicedwass_expt_{experiment_identifier}_{gen_method}.csv', index=False)

    elif metric == 'mmd':
        # Compute the MMD distance between the source and the generated datasets
        mmd_distances = []
        for gen_dataset in gen_datasets:
            mmd_distance = compute_mmd_distance(source_df, gen_dataset)
            mmd_distances.append(mmd_distance)

        # Save the MMD distances to a csv file
        mmd_distances_df = pd.DataFrame(mmd_distances, columns=['mmd_distance'])
        mmd_distances_df.to_csv(f'{METRICS_PATH}/mmd_expt_{experiment_identifier}_{gen_method}.csv',
                                index=False)
    else:
        raise ValueError(f'Metric {metric} not implemented')


# Main function
if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_identifier', type=str, required=True)
    args = parser.parse_args()

    # for gen_method in ['realcause', 'credence', 'mcredence', 'frugalflows']:
    #     compute_metrics(args.experiment_identifier, gen_method, metric='mmd')
    #     compute_metrics(args.experiment_identifier, gen_method, metric='slicedwass')

    # For each experiment identifier and generative methods, write code to compute the mean
    # of the metrics per generative method and put it together into a single dataframe
    mean_metrics = pd.DataFrame(
        columns=['gen_method', 'mean_mmd', 'mean_slicedwass', 'std_mmd', 'std_slicedwass'])
    for gen_method in ['realcause', 'credence', 'mcredence', 'frugalflows']:
        mmd_distances = pd.read_csv(
            f'{METRICS_PATH}/mmd_expt_{args.experiment_identifier}_{gen_method}.csv')
        slicedwass_distances = pd.read_csv(
            f'{METRICS_PATH}/slicedwass_expt_{args.experiment_identifier}_{gen_method}.csv')
        # Compute the mean per generative method and the standard deviation of the metrics
        mean_mmd = mmd_distances['mmd_distance'].mean()
        mean_slicedwass = slicedwass_distances['slicedwass_distance'].mean()
        std_mmd = mmd_distances['mmd_distance'].std()
        std_slicedwass = slicedwass_distances['slicedwass_distance'].std()
        # Add a column for the generative method and add the corresponding mean metrics
        mean_metrics = pd.concat([
            mean_metrics,
            pd.DataFrame([{
                'gen_method': gen_method,
                'mean_mmd': mean_mmd,
                'mean_slicedwass': mean_slicedwass,
                'std_mmd': std_mmd,
                'std_slicedwass': std_slicedwass
            }])
        ],
                                 ignore_index=True)

    # Save the mean metrics to a csv file
    mean_metrics.to_csv(f'{METRICS_PATH}/mean_metrics_expt_{args.experiment_identifier}.csv',
                        index=False)
