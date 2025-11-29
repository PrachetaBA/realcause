# pylint: disable=import-error, logging-fstring-interpolation, possibly-used-before-assignment
"""Script to run the ATE estimators on the generated datasets when there
are two possible models: Realcause and FrugalFlows.

Conda environment: /work/pi_jensen_umass_edu/pboddavarama_umass_edu/pba-conda/envs/rpy

The user may specify the ATE estimators as well as the range
of replications to run. This will help in parallelizing the
jobs, especially for large sample sizes.

This is an updated version of the script `run_causal_estimators_v2.py
in the nfl-causal-estimation repository.
"""

# Import libraries
import argparse
import logging
import os

import pandas as pd
from . import ate_estimators
from loading import load_gen
from data_loaders import apo, lalonde, twins

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
ESTIMATED_ATE_PATH = 'output/ate_estimates_models'


def load_source_dataset(dataset_name,
                        dataset_identifier=None,
                        sample_size=None,
                        dataset_path=None,
                        realcause_model_path=None):
    """Function to load the base datasets to run the causal estimators
    and find the true ATE of the dataset.

    Args:
        dataset_name: str
            Name of the dataset.
        dataset_identifier: str
            Identifier of the dataset.
        sample_size: int
            Sample size of the dataset.
        realcause_model_path: str
            Path to the Realcause model used to standardize the dataset (and then compute the true ATE).
    Returns:
        true_ate: float
            True ATE of the dataset (if known)
        outcome: str
            Outcome variable of the dataset.
        treatment: str
            Treatment variable of the dataset.
        data: pd.DataFrame
            Dataset with all the variables.
    """
    # Load the observed dataset to be used as the reference dataset
    if dataset_name in ['n_acic_4', 'jdk', 'postgres']:
        d = apo.get_apo_data(identifier=dataset_name,
                             confound_func=dataset_identifier,
                             data_format='pandas',
                             return_ites=True,
                             ret_counterfactual_outcomes=False,
                             sample_size=sample_size)
        # Get a pandas dataframe from the combination of the orig columns
        true_ate = d['ite'].mean()
        treatment_col = d['t'].name    # Get only the name of the treatment column
        outcome_col = d['y'].name    # Get only the name of the outcome column
    elif dataset_name == 'lalonde':
        if dataset_identifier == 'psid1':
            d = lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
        elif dataset_identifier == 'cps1':
            d = lalonde.load_lalonde(obs_version='cps', data_format='pandas_single')
        d.drop(columns=['data_id'], inplace=True)
        outcome_col = 're78'
        treatment_col = 'treat'
        covariates_col = d.columns.tolist()
        covariates_col.remove(outcome_col)
        covariates_col.remove(treatment_col)
        # Reorder the columns to put all the covariates first, then treatment, then outcome
        d = d[covariates_col + [treatment_col, outcome_col]]
        # Compute the true ATE as just the values from the RCT data (after applying the standardization from the Realcause model)
        rc_model, _ = load_gen(saveroot=realcause_model_path)
        # Apply the transformmations to the source data and compute the true ATE
        rct_data = lalonde.load_lalonde(rct_version='dw', rct=True, data_format='pandas_single')
        rct_data[covariates_col] = rc_model.w_transform.transform(rct_data[covariates_col].values)
        rct_data[outcome_col] = rc_model.y_transform.transform(rct_data[outcome_col].values.reshape(
            -1, 1))
        true_ate = rct_data['re78'][rct_data['treat'] == 1].mean() - rct_data['re78'][
            rct_data['treat'] == 0].mean()
    elif dataset_name == 'twins':
        d = twins.load_twins(data_format='pandas', return_sketchy_ites=True)
        treatment_col = 'T'
        outcome_col = 'yf'
        true_ate = d['ites'].mean()
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented')
    source_data = pd.read_csv(f'{dataset_path}/observed.csv')
    return {
        'true_ate': true_ate,
        'outcome': outcome_col,
        'treatment': treatment_col,
        'data': source_data
    }


def load_smcabc(dataset_name,
                dataset_identifier,
                sample_size,
                dataset_path,
                dataset_number,
                observed_data=False,
                posterior_or_prior=None,
                realcause_model_path=None):
    """Function to load the SMC-ABC datasets using Realcause and FrugalFlows as the simulator models.

    Args:
        dataset_name: str
            Name of the dataset.
        dataset_identifier: str
            Identifier of the dataset.
        sample_size: int
            Sample size of the dataset.
        dataset_path: str
            Path to the Realcause model.
        dataset_number: int
            Iteration number of the dataset.
        observed_data: pd.DataFrame
            Source dataset that is used to generate the Realcause/FrugalFlows datasets
        posterior_or_prior: str
            if 'posterior', load the posterior dataset, else load the prior dataset.
        realcause_model_path: str
            Path to the Realcause model used to standardize the dataset (and then compute the true ATE).

    Returns:
        parameters: dict
            Dictionary of the parameters of the dataset.
        data: pd.DataFrame
            Dataset with all the variables.
    """
    parameters_dict = {}
    output = load_source_dataset(dataset_name=dataset_name,
                                 dataset_identifier=dataset_identifier,
                                 sample_size=sample_size,
                                 dataset_path=dataset_path,
                                 realcause_model_path=realcause_model_path)
    parameters_dict['te'] = output['true_ate']

    if not observed_data:
        # Load the parameter samples for the Realcause model
        # Rewrite the data and the parameter values for this
        rc_parameters = pd.read_csv(f'{dataset_path}/rc_parameter_samples.csv')
        rc_data = pd.DataFrame()
        # Proceed only if rc_parameters is not empty
        if not rc_parameters.empty:
            # Map parameter prefixes and their corresponding output keys
            prefix = 'post_' if posterior_or_prior == 'posterior' else 'prior_'
            param_mapping = {
                f'{prefix}te': 'te',
                f'{prefix}deg_hetero': 'deg_hetero',
                f'{prefix}overlap': 'overlap'
            }

            # Load dataset file
            rc_data = pd.read_csv(
                f'{dataset_path}/rc_{posterior_or_prior}_sample_{dataset_number}.csv')

            # Extract row once for efficiency
            param_row = rc_parameters.iloc[dataset_number]

            # Extract all parameters in a single pass
            for param_col, output_key in param_mapping.items():
                if param_col in rc_parameters.columns:
                    parameters_dict[output_key] = param_row[param_col]

        # Load the parameter samples for the FrugalFlows model
        ff_parameters = pd.read_csv(f'{dataset_path}/ff_parameter_samples.csv')
        ff_data = pd.DataFrame()
        # Proceed only if ff_parameters is not empty
        if not ff_parameters.empty:
            # Load dataset file
            ff_data = pd.read_csv(
                f'{dataset_path}/ff_{posterior_or_prior}_sample_{dataset_number}.csv')
            # Map parameter prefixes and their corresponding output keys
            prefix = 'post_' if posterior_or_prior == 'posterior' else 'prior_'
            param_mapping = {
                f'{prefix}ate': 'te',
                f'{prefix}rho': 'rho',
            }

            # Extract row once for efficiency
            param_row = ff_parameters.iloc[dataset_number]
            # Extract all parameters in a single pass
            for param_col, output_key in param_mapping.items():
                if param_col in ff_parameters.columns:
                    parameters_dict[output_key] = param_row[param_col]
    # Combine the rc_data and ff_data into a single dataframe
    data = pd.concat([rc_data, ff_data], axis=0)
    return {
        'parameters': parameters_dict,
        'outcome_col': output['outcome'],
        'treatment_col': output['treatment'],
        'data': data
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run causal estimators on the Realcause datasets.')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--dataset_identifier', type=str, default=None)
    parser.add_argument('--sample_size', type=str, default=None, required=False)
    parser.add_argument('--experiment_number', type=int, default=None)
    parser.add_argument('--distance_function', type=str, default='sliced_wass', required=False)
    parser.add_argument('--observed_data', action='store_true', default=False)
    parser.add_argument('--posterior_or_prior',
                        type=str,
                        default='posterior',
                        required=False,
                        choices=['posterior', 'prior'])
    parser.add_argument('--num_replications', type=int, default=50, required=False)
    parser.add_argument('--set_of_estimators', type=str, default='all', required=False)
    parser.add_argument('--realcause_model_path', type=str, default=None, required=False)
    parser.add_argument('--smc_expt_id', type=int, default=None, required=True)
    # For e.g. python -m sbice.get_ate_estimates --dataset_name lalonde --dataset_identifier psid1
    # --experiment_number 3 --posterior_or_prior posterior --num_replications 1 --set_of_estimators meta
    # --realcause_model_path results/GenModelCkpts/lalonde/psid1/save
    args = parser.parse_args()
    logger.info(f'Arguments: {args}')

    if args.observed_data:
        source_data_info = load_source_dataset(args.dataset_name,
                                               dataset_identifier=args.dataset_identifier,
                                               sample_size=args.sample_size,
                                               realcause_model_path=args.realcause_model_path)
        # Use the observed data to compute the ATE estimates on the source data
        logger.info(
            f'Running ATE estimators on {args.dataset_name}_{args.dataset_identifier}_{args.sample_size} dataset!'
        )
        logger.info(f'True ATE (transformed, if applicable): {source_data_info["true_ate"]}')
        # Use this to extract the true ATE from the source data (after applying the transformations if applicable)
        source_data_true_ate = source_data_info['true_ate']
        # Load the observed data
        observed_data_info = load_source_dataset(
            dataset_name=args.dataset_name,
            dataset_identifier=args.dataset_identifier,
            sample_size=args.sample_size,
            dataset_path=
            f'data/smc_abc/{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}_dist_{args.distance_function}_expt_{args.experiment_number}/{args.smc_expt_id}',
            realcause_model_path=args.realcause_model_path)
        estimated_ate = ate_estimators.bootstrap_ate_inference(
            outcome=observed_data_info['outcome_col'],
            treatment=observed_data_info['treatment_col'],
            data=observed_data_info['data'],
            dataset_identifier=
            f'{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}_expt_{args.experiment_number}_base',
            set_of_estimators=ALL_ESTIMATORS
            if args.set_of_estimators == 'all' else META_ESTIMATORS,
            repeats=1)
        estimated_ate['df'] = f'{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}'
        estimated_ate['true_ate'] = source_data_true_ate

        ate_df_path = f'{ESTIMATED_ATE_PATH}/{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}/'
        if not os.path.exists(ate_df_path):
            os.makedirs(ate_df_path)
        logger.info(f'Computed ATEs for the source dataset!')
        # Save the dataframe to a csv file
        # Experiment number is required to be specified (in case the transformation is applied to the source data)
        ATE_DF_FILENAME = f'{ate_df_path}{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}_expt_{args.experiment_number}_base_ate.csv'
        logger.info(f'Saving the ATE estimates to {ATE_DF_FILENAME}')
        estimated_ate.to_csv(ATE_DF_FILENAME, index=False)

    else:
        ate_df = pd.DataFrame()
        logger.info(
            f'Running ATE estimators on {args.dataset_name}_{args.dataset_identifier}_{args.sample_size} dataset!'
        )
        for itr in range(args.num_replications):
            logger.info(f'Iteration {itr}')
            gen_data_info = load_smcabc(
                dataset_name=args.dataset_name,
                dataset_identifier=args.dataset_identifier,
                sample_size=args.sample_size,
                dataset_path=
                f'data/smc_abc/{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}_dist_{args.distance_function}_expt_{args.experiment_number}/{args.smc_expt_id}',
                dataset_number=itr,
                observed_data=False,
                posterior_or_prior=args.posterior_or_prior,
                realcause_model_path=args.realcause_model_path)
            # Run the ATE estimators on all the datasets
            estimated_ate = ate_estimators.bootstrap_ate_inference(
                outcome=gen_data_info['outcome_col'],
                treatment=gen_data_info['treatment_col'],
                data=gen_data_info['data'],
                dataset_identifier=
                f'{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}_dist_{args.distance_function}_expt_{args.experiment_number}_{itr}',
                set_of_estimators=ALL_ESTIMATORS
                if args.set_of_estimators == 'all' else META_ESTIMATORS,
                repeats=1)
            estimated_ate[
                'df'] = f'{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}'
            estimated_ate['true_ate'] = gen_data_info['parameters']['te']
            logger.info(f'ATE estimate: {gen_data_info["parameters"]["te"]}')
            ate_df = pd.concat([ate_df, estimated_ate], axis=0)

        logger.info(f'#' * 50)
        ate_df_path = f'{ESTIMATED_ATE_PATH}/{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}/'
        if not os.path.exists(ate_df_path):
            os.makedirs(ate_df_path)
        logger.info(f'Computed ATEs for the generated dataset!')
        # Save the dataframe to a csv file
        ATE_DF_FILENAME = f'{ate_df_path}{args.dataset_name}_{args.dataset_identifier}_{args.sample_size}_expt_{args.experiment_number}_{args.posterior_or_prior}_ate.csv'
        logger.info(f'Saving the ATE estimates to {ATE_DF_FILENAME}')
        ate_df.to_csv(ATE_DF_FILENAME, index=False)
