# pylint: disable=import-error, logging-fstring-interpolation, possibly-used-before-assignment
"""Script to run the ATE estimators on the generated datasets.

Conda environment: rpy

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
from tqdm import tqdm

# from . import ate_estimators  #TEMPOrarily commented out
from data_loaders import apo, lalonde, twins

# Set logger to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_source_dataset(dataset_name, 
                        dataset_identifier=None,
                        sample_size=None):
    """Function to load the base datasets to run the causal estimators
    and find the true ATE of the dataset.

    Args:
        dataset_name: str
            Name of the dataset.
        dataset_identifier: str
            Identifier of the dataset.
        sample_size: int
            Sample size of the dataset.

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
        d = apo.get_apo_data(identifier=dataset_name, confound_func=dataset_identifier, 
                             data_format='pandas', return_ites=True, 
                             ret_counterfactual_outcomes=False,
                             sample_size=sample_size)
        # Get a pandas dataframe from the combination of the orig columns
        source_data = pd.concat([d['w'], d['t'], d['y'], d['ite']], axis=1)
        true_ate = d['ite'].mean()
        treatment_col = d['t'].name  # Get only the name of the treatment column
        outcome_col = d['y'].name  # Get only the name of the outcome column
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
        covariates_df = d[covariates_col].values
        source_data = d
        # Compute the true ATE as just the values from the RCT data
        rct_data = lalonde.load_lalonde(rct_version='dw', rct=True, data_format='pandas_single')
        true_ate = rct_data['re78'][rct_data['treat'] == 1].mean() - rct_data['re78'][rct_data['treat'] == 0].mean()
    elif dataset_name == 'twins':
        d = twins.load_twins(data_format='pandas', return_sketchy_ites=True)
        source_data = pd.concat([d['w'], d['t'], d['y'], d['ites']], axis=1)
        covariates_df = d['w'].values
        treatment_col = 'T'
        outcome_col = 'yf'
        true_ate = d['ites'].mean() 
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    
    return {
        'true_ate': true_ate,
        'outcome': outcome_col,
        'treatment': treatment_col,
        'data': source_data
    }


if __name__ == '__main__':
    # TEST source dataset loading
    info = load_source_dataset('lalonde', 'psid1')
    print(info)
    info = load_source_dataset('twins')
    print(info)
    info = load_source_dataset('lalonde', 'cps1')
    print(info) 
    info = load_source_dataset('postgres', 'linear', 3000)
    print(info)
'''
def load_smcabc(data_id, path, itr, posterior=True, observed_dist=False, drop_unobs_cov=False):
    """Function to load the SMC-ABC datasets using FrugalFlows as the simulator model.

    Args:
        data_id: str
            Identifier of the dataset.
        path: str
            Path to the SMC-ABC dataset.
        itr: int
            Iteration number of the dataset.
        posterior: bool
            if true, load the posterior dataset, else load the prior dataset.
        observed_dist: bool
            if true, load the observed distribution dataset (not posterior or prior).

    Returns:
        true_ate: float
            True ATE of the dataset.
        outcome: str
            Outcome variable of the dataset.
        treatment: str
            Treatment variable of the dataset.
        data: pd.DataFrame
            Dataset with all the variables.
    """
    true_rho = None
    # We need to extract the corresponding ATE for each of the iterations only for the posterior and prior
    if not observed_dist:
        parameters = pd.read_csv(f'{path}parameter_samples.csv')

    if posterior:
        df = pd.read_csv(f'{path}posterior_sample_{itr}.csv')
        if data_id == 'lalonde':
            outcome_col = 're78'
            treatment_col = 'treat'
        elif data_id == 'syn_unobs':
            if any(x in path for x in ['dgp1', 'dgp2', 'dgp4', 'dgp5', 'dgp6']):
                outcome_col = 'Y'
                treatment_col = 'T'
            elif any(x in path for x in ['dgp7']):
                outcome_col = 'y'
                treatment_col = 't'
        elif data_id == 'project_star':
            outcome_col = 'g3avgscore'
            treatment_col = 'g3smallclass'
        elif data_id.startswith('frugal_param'):
            outcome_col = 'Y'
            treatment_col = 'T'
        elif data_id.startswith('causl'):
            outcome_col = 'Y'
            treatment_col = 'T'
        true_ate = parameters['post_ate'].iloc[itr]
    elif posterior is False and observed_dist is False:
        df = pd.read_csv(f'{path}prior_sample_{itr}.csv')
        if data_id == 'lalonde':
            outcome_col = 're78'
            treatment_col = 'treat'
        elif data_id == 'syn_unobs':
            if any(x in path for x in ['dgp1', 'dgp2', 'dgp4', 'dgp5', 'dgp6']):
                outcome_col = 'Y'
                treatment_col = 'T'
            elif any(x in path for x in ['dgp7']):
                outcome_col = 'y'
                treatment_col = 't'
        elif data_id == 'project_star':
            outcome_col = 'g3avgscore'
            treatment_col = 'g3smallclass'
        elif data_id.startswith('frugal_param'):
            outcome_col = 'Y'
            treatment_col = 'T'
        elif data_id.startswith('causl'):
            outcome_col = 'Y'
            treatment_col = 'T'
        true_ate = parameters['prior_ate'].iloc[itr]
    elif posterior is False and observed_dist is True:
        if data_id == 'syn_unobs':
            df = pd.read_csv(f'{path}observed_{itr}.csv')
            if any(x in path for x in ['dgp4', 'dgp6']):
                outcome_col = 'Y'
                treatment_col = 'T'
                # Drop the following columns: Z1 and Z2
                df.drop(columns=['Z1', 'Z2'], inplace=True)
                # Reorder the dataset to have the following order: Y, T, X1, X2, X3
                df = df[['Y', 'T', 'X1', 'X2', 'X3']]
                true_ate = 3.0    # Has to be set manually, true rho is a matrix.
            elif any(x in path for x in ['dgp7']):
                outcome_col = 'y'
                treatment_col = 't'
                # Drop the following columns: z, y_cf, y1, y0, ite
                df.drop(columns=['z', 'y_cf'], inplace=True)
                # Reorder the dataset to have the following order: y, t, x1, x2, x3
                df = df[['y', 't', 'x']]
                true_ate = 3.0    # Set manually, true rho is unknown
        elif data_id.startswith('frugal_param'):
            outcome_col = 'Y'
            treatment_col = 'T'
            if 'dgp1' in path and 'small' not in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp1_{itr}.csv')
                    df.drop(columns=['X2'], inplace=True)
                    true_ate = 5.0
                    true_rho = -0.3
                else:
                    df = pd.read_csv(f'{path}frugal_dgp1_{itr}.csv')
                    true_ate = 5.0
                    true_rho = 0.0
            elif 'dgp1_small' in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp1_small_{itr}.csv')
                    df.drop(columns=['X2'], inplace=True)
                    true_ate = 5.0
                    true_rho = -0.3
                else:
                    df = pd.read_csv(f'{path}frugal_dgp1_small_{itr}.csv')
                    true_ate = 5.0
                    true_rho = 0.0
            elif 'dgp2' in path and 'small' not in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp2_{itr}.csv')
                    df.drop(columns=['X4'], inplace=True)
                    true_ate = 5.0
                    true_rho = 0.8
                else:
                    df = pd.read_csv(f'{path}frugal_dgp2_{itr}.csv')
                    true_ate = 5.0
                    true_rho = 0.0
            elif 'dgp2_small' in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp2_small_{itr}.csv')
                    df.drop(columns=['X4'], inplace=True)
                    true_ate = 5.0
                    true_rho = 0.8
                else:
                    df = pd.read_csv(f'{path}frugal_dgp2_small_{itr}.csv')
                    true_ate = 5.0
                    true_rho = 0.0
            elif 'dgp3' in path and 'small' not in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp3_{itr}.csv')
                    df.drop(columns=['X3', 'X7'], inplace=True)
                    true_ate = -5.0
                else:
                    df = pd.read_csv(f'{path}frugal_dgp3_{itr}.csv')
                    true_ate = -5.0
                    true_rho = 0.0
            elif 'dgp3_small' in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp3_small_{itr}.csv')
                    df.drop(columns=['X3', 'X7'], inplace=True)
                    true_ate = -5.0
                else:
                    df = pd.read_csv(f'{path}frugal_dgp3_small_{itr}.csv')
                    true_ate = -5.0
                    true_rho = 0.0
            elif 'dgp4' in path:
                if drop_unobs_cov:
                    df = pd.read_csv(f'{path}frugal_dgp4_{itr}.csv')
                    df.drop(columns=['X2'], inplace=True)
                    true_ate = -2.0
                    true_rho = -0.5
                else:
                    df = pd.read_csv(f'{path}frugal_dgp4_{itr}.csv')
                    true_ate = -2.0
                    true_rho = 0.0
        elif data_id.startswith('causl'):
            outcome_col = 'Y'
            treatment_col = 'T'
            if 'dgp1' in path:
                df = pd.read_csv(f'{path}frugal_dgp1_{itr}.csv')
                true_ate = 5.0
                if drop_unobs_cov:
                    df.drop(columns=['X2'], inplace=True)
            elif 'dgp2' in path:
                df = pd.read_csv(f'{path}frugal_dgp2_{itr}.csv')
                true_ate = 5.0
                if drop_unobs_cov:
                    df.drop(columns=['X4'], inplace=True)
            elif 'dgp3' in path:
                df = pd.read_csv(f'{path}frugal_dgp3_{itr}.csv')
                true_ate = -5.0
            elif 'dgp4' in path:
                df = pd.read_csv(f'{path}frugal_dgp4_{itr}.csv')
                true_ate = -2.0
            elif 'dgp5' in path:
                df = pd.read_csv(f'{path}frugal_dgp5_{itr}.csv')
                true_ate = -10.0
            elif 'dgp6' in path:
                df = pd.read_csv(f'{path}frugal_dgp6_{itr}.csv')
                true_ate = -1.5
    return {
        'true_ate': true_ate,
        'outcome': outcome_col,
        'treatment': treatment_col,
        'data': df,
        'true_rho': true_rho
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run causal estimators on the Credence/RealCause/WGAN/FrugalFlows datasets.')
    args = argparse.ArgumentParser()
    args.add_argument('--dataset',
                      '-d',
                      type=str,
                      default=None,
                      choices=[
                          'lalonde',
                          'apo_acic',
                          'kunzel',
                          'syn_linear',
                          'syn_unobs',
                          'project_star',
                          'frugal_param',
                          'causl',
                          'e401k'
                      ])
    args.add_argument(
        '--dataset_identifier',
        '-di',
        type=str,
        default=None,
    )
    args.add_argument('--base_or_gen',
                      '-bg',
                      type=str,
                      default='gen',
                      required=False,
                      choices=['base', 'gen'])
    args.add_argument(
        '--gen_method',
        '-g',
        type=str,
        default=None,
        required=False,
        choices=['realcause', 'credence', 'modified_credence', 'wgan', 'frugalflows', 'smcabc'])
    args.add_argument('--num_replications',
                      '-n',
                      type=int,
                      default=50,
                      required=False,
                      help='Number of replications to use.')
    args.add_argument(
        '--experiment_identifier',
        '-ei',
        type=str,
        default=None,
        required=False,
        help=
        'Identifier for the experiment; Used when the dataset is `syn_linear` or `syn_unobs` and the method is Credence.'
    )
    args.add_argument(
        '--posterior_or_prior',
        '-pp',
        type=str,
        default='posterior',
        required=False,
        help='Whether to use the posterior or prior samples for the SMC-ABC datasets.',
        choices=['posterior', 'prior', 'observed_dist'])

    args = args.parse_args()

    # Constants that are used in the script to define the set of
    # estimators that will be run.
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

    if args.base_or_gen == 'base':
        if args.dataset == 'lalonde':
            if args.dataset_identifier == 'exp':
                dataset_info = load_base_dataset('lalonde_exp')
            elif args.dataset_identifier == 'obs':
                dataset_info = load_base_dataset('lalonde_obs')
        elif args.dataset == 'apo_acic':
            dataset_info = load_base_dataset('apo_acic')
        elif args.dataset == 'kunzel':
            dataset_info = load_base_dataset(f'{args.dataset}_{args.dataset_identifier}')
        elif args.dataset == 'syn_linear':
            dataset_info = load_base_dataset(f'{args.dataset}_{args.dataset_identifier}')
        elif args.dataset == 'syn_unobs':
            dataset_info = load_base_dataset(f'{args.dataset}_{args.dataset_identifier}')
        elif args.dataset == 'project_star':
            dataset_info = load_base_dataset(f'{args.dataset}_{args.dataset_identifier}')
        elif args.dataset == 'frugal_param':
            dataset_info = load_base_dataset(f'{args.dataset}_{args.dataset_identifier}')
        elif args.dataset == 'e401k':
            dataset_info = load_base_dataset()
        else:
            raise ValueError('Unknown dataset!')

        logger.info(f'Running ATE estimators on {args.dataset} dataset!')

        estimated_ate = ate_estimators.bootstrap_ate_inference(
            outcome=dataset_info['outcome'],
            treatment=dataset_info['treatment'],
            data=dataset_info['data'],
            dataset_identifier=f'{args.dataset}_{args.dataset_identifier}',
            set_of_estimators=ALL_ESTIMATORS,
            repeats=1)
        if args.dataset_identifier:
            estimated_ate['df'] = f'{args.dataset}_{args.dataset_identifier}'
        else:
            estimated_ate['df'] = args.dataset
        estimated_ate['true_ate'] = dataset_info['true_ate']

        ate_df_path = f'{ESTIMATED_ATE_PATH}ate_estimates_{args.dataset}/'
        if not os.path.exists(ate_df_path):
            os.makedirs(ate_df_path)

        logger.info(f'Computed ATEs for the base dataset!')

        # Save the dataframe to a csv file
        if args.dataset_identifier:
            ATE_DF_FILENAME = f'{ate_df_path}{args.dataset}_{args.dataset_identifier}_base_ate.csv'
        else:
            ATE_DF_FILENAME = f'{ate_df_path}{args.dataset}_base_ate.csv'
        logger.info(f'Saving the ATE estimates to {ATE_DF_FILENAME}')
        estimated_ate.to_csv(ATE_DF_FILENAME, index=False)

    elif args.base_or_gen == 'gen':
        if args.gen_method == 'credence':
            if args.dataset == 'lalonde':
                # OLD - DELETE
                # DATA_SUBFOLDER = f'{VMK_DATA_PATH}credence_{args.dataset}/{args.dataset_identifier}/'
                DATA_SUBFOLDER = f'data/credence_{args.dataset}_{args.dataset_identifier}/{args.dataset}_{args.dataset_identifier}_{args.experiment_identifier}/'
            elif args.dataset == 'apo_acic':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}credence_acic/'
            elif args.dataset == 'kunzel':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}credence_{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'syn_linear':
                DATA_SUBFOLDER = f'data/credence_synthetic/synthetic_{args.experiment_identifier}/'
            elif args.dataset == 'syn_unobs':
                # For e.g. for dgp7, we have dataset_identifier = 7 and experiment_identifier = 'synthetic_unobserved_7_flexiblecb'
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}credence_synthetic_unobserved/synthetic_unobserved_{args.dataset_identifier}/{args.experiment_identifier}/'
        elif args.gen_method == 'modified_credence':
            if args.dataset == 'lalonde':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}modified_credence_{args.dataset}/{args.dataset_identifier}/{args.experiment_identifier}/'
            elif args.dataset == 'apo_acic':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}modified_credence_acic/'
            elif args.dataset == 'kunzel':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}modified_credence_{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'syn_linear':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}modified_credence_synthetic_simple/{args.dataset_identifier}_{args.experiment_identifier}/'
            elif args.dataset == 'syn_unobs':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}modified_credence_synthetic_unobserved/synthetic_unobserved_{args.dataset_identifier}/{args.experiment_identifier}/'
        elif args.gen_method == 'realcause':
            if args.dataset == 'lalonde':
                DATA_SUBFOLDER = f'{REALCAUSE_DATA_PATH}{args.dataset}_{args.dataset_identifier}/'
                # OLD - DELETE
                # if args.dataset_identifier == 'exp':
                #     DATA_SUBFOLDER = f'{REALCAUSE_DATA_PATH}{args.dataset}_exp_{args.dataset_identifier}/'
                # elif args.dataset_identifier == 'obs':
                #     DATA_SUBFOLDER = f'{REALCAUSE_DATA_PATH}{args.dataset}_obs_{args.dataset_identifier}/'
            elif args.dataset == 'apo_acic':
                DATA_SUBFOLDER = f'{REALCAUSE_DATA_PATH}osapo_acic_4_weight_1_intercept_0/'
            elif args.dataset == 'kunzel':
                DATA_SUBFOLDER = f'{REALCAUSE_DATA_PATH}{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'syn_linear':
                DATA_SUBFOLDER = f'{REALCAUSE_DATA_PATH}{args.dataset}_{args.dataset_identifier}/'
        elif args.gen_method == 'wgan':
            if args.dataset == 'lalonde':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}wgan_{args.dataset}/{args.dataset_identifier}/'
            elif args.dataset == 'apo_acic':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}wgan_acic_best/'
            elif args.dataset == 'kunzel':
                DATA_SUBFOLDER = f'{VMK_DATA_PATH}wgan_{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
        elif args.gen_method == 'frugalflows':
            if args.dataset == 'lalonde':
                DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
                # if args.dataset_identifier == 'dw':
                # DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}{args.dataset}_exp/'
                # elif args.dataset_identifier == 'psid':
                # DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}{args.dataset}_obs/'
            elif args.dataset == 'apo_acic':
                DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}acic/osapo_acic_4/'
            elif args.dataset == 'kunzel':
                if args.dataset_identifier == '2_ss_2000':
                    DATA_SUBFOLDER = f'{VMK_FRUGALFLOWS_DATA_PATH}{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
                elif args.dataset_identifier == '4_ss_2000':
                    DATA_SUBFOLDER = f'{VMK_FRUGALFLOWS_DATA_PATH}{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
                elif args.dataset_identifier == '5_ss_2000':
                    DATA_SUBFOLDER = f'{VMK_FRUGALFLOWS_DATA_PATH}{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
                elif args.dataset_identifier == '6_ss_2000':
                    DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}{args.dataset}/{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'syn_linear':
                DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}synthetic/synthetic_{args.dataset_identifier}/'
            elif args.dataset == 'syn_unobs':
                DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}synthetic_unobs/synthetic_unobs_{args.dataset_identifier}/'
            elif args.dataset == 'frugal_param':
                DATA_SUBFOLDER = f'{FRUGALFLOWS_DATA_PATH}frugal_param/{args.dataset}_{args.dataset_identifier}/'
        elif args.gen_method == 'smcabc':
            if args.dataset == 'lalonde':
                if 'exp_dist_sliced_wass_expt_42' in args.dataset_identifier:
                    DATA_SUBFOLDER = f'{SMCABC_VMK_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                else:
                    DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'syn_unobs':
                DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'project_star':
                DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
            elif args.dataset == 'frugal_param':
                if args.posterior_or_prior in ['posterior', 'prior']:
                    if 'dgp1' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                    elif 'dgp2' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                    elif 'dgp2_unobs' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                    elif 'dgp3' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                    elif 'dgp4' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{SMCABC_VMK_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                elif args.posterior_or_prior == 'observed_dist':
                    if 'dgp1' in args.dataset_identifier and 'small' not in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp1/'
                    elif 'dgp1_small' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp1_small/'
                    elif 'dgp2' in args.dataset_identifier and 'small' not in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp2/'
                    elif 'dgp2_small' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp2_small/'
                    elif 'dgp3' in args.dataset_identifier and 'small' not in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp3/'
                    elif 'dgp3_small' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp3_small/'
                    elif 'dgp4' in args.dataset_identifier:
                        DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/dgp4/'
            elif args.dataset == 'causl':
                if args.posterior_or_prior in ['posterior', 'prior']:
                    DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
                elif args.posterior_or_prior == 'observed_dist':
                    DATA_SUBFOLDER = f'{RGEN_DATA_PATH}/{args.dataset_identifier.split("_")[0]}/'
            elif args.dataset == 'e401k':
                DATA_SUBFOLDER = f'{SMCABC_DATA_PATH}/{args.dataset}_{args.dataset_identifier}/'
        else:
            raise ValueError('Unknown dataset!')

        logger.info(f'Data subfolder is: {DATA_SUBFOLDER}')

        # Create the folder to save the ATE estimates
        ate_df_path = f'{ESTIMATED_ATE_PATH}ate_estimates_{args.dataset}/'
        if not os.path.exists(ate_df_path):
            os.makedirs(ate_df_path)

        logger.info(f'Running ATE estimators on {args.dataset} dataset using {args.gen_method}!')

        updated_dataset_identifier = args.dataset_identifier
        # Create the empty dataframe
        ate_df = pd.DataFrame()
        for itr in tqdm(range(args.num_replications)):
            logger.info(f'Iteration: {itr}')
            if args.gen_method == 'credence':
                dataset_info = load_credence(args.dataset, DATA_SUBFOLDER, itr)
                if args.dataset_identifier == '7':
                    if args.experiment_identifier == 'synthetic_unobserved_7_flexiblecb':
                        updated_dataset_identifier = 'dgp7'
            elif args.gen_method == 'modified_credence':
                dataset_info = load_modified_credence(args.dataset, DATA_SUBFOLDER, itr)
                if args.dataset_identifier == '7':
                    if args.experiment_identifier == 'synthetic_unobserved_7_flexiblecb':
                        updated_dataset_identifier = 'dgp7'
            elif args.gen_method == 'wgan':
                dataset_info = load_wgan(args.dataset, DATA_SUBFOLDER, itr)
            elif args.gen_method == 'realcause':
                dataset_info = load_realcause(args.dataset, DATA_SUBFOLDER, itr)
            elif args.gen_method == 'frugalflows':
                if args.dataset == 'syn_linear':
                    if 'ate' in args.dataset_identifier:
                        ate_val = args.dataset_identifier.split('_')[-1]
                    else:
                        ate_val = 3.002091512943254
                    dataset_info = load_frugalflows(args.dataset, DATA_SUBFOLDER, itr, ate=ate_val)
                elif args.dataset == 'syn_unobs':
                    ate_val = 3.0    # dgp1, dgp2, dgp4, dgp6 and dgp7 all have ate_val = 3.0
                    dataset_info = load_frugalflows(args.dataset, DATA_SUBFOLDER, itr, ate=ate_val)
                elif args.dataset == 'lalonde':
                    if 'rho' in args.dataset_identifier and 'ate' in args.dataset_identifier:
                        # Extract the ate value which is lies between _ and _ immediately following 'ate'
                        # e.g. dataset_identifier = 'exp_ate_0.27057904_rho_0.1'
                        ate_val = args.dataset_identifier.split('_')[-2]
                    elif 'ate' in args.dataset_identifier and 'rho' not in args.dataset_identifier:
                        ate_val = args.dataset_identifier.split('_')[-1]
                    else:
                        ate_val = 0.27057904
                    dataset_info = load_frugalflows(args.dataset, DATA_SUBFOLDER, itr, ate=ate_val)
                elif args.dataset == 'kunzel':
                    if args.dataset_identifier in [
                            '2_ss_2000', '4_ss_2000', '5_ss_2000', '6_ss_2000'
                    ]:
                        dataset_info = load_frugalflows(args.dataset, DATA_SUBFOLDER, itr)
                elif args.dataset == 'frugal_param':
                    if 'dgp1' in args.dataset_identifier:
                        if 'rho' in args.dataset_identifier and 'ate' not in args.dataset_identifier:
                            rho_val = args.dataset_identifier.split('_')[-1]
                            ate_val = 5.054540647411587    # Learned ATE for dgp1
                        elif 'ate' in args.dataset_identifier and 'rho' not in args.dataset_identifier:
                            ate_val = args.dataset_identifier.split('_')[-1]
                        elif 'ate' in args.dataset_identifier and 'rho' in args.dataset_identifier:
                            rho_val = args.dataset_identifier.split('_')[-1]
                            ate_val = args.dataset_identifier.split('_')[-3]
                        else:
                            ate_val = 5.054540647411587    # Learned ATE for dgp1
                    dataset_info = load_frugalflows(args.dataset, DATA_SUBFOLDER, itr, ate=ate_val)
            elif args.gen_method == 'smcabc':
                if args.dataset == 'lalonde':
                    if args.posterior_or_prior == 'posterior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=True)
                    elif args.posterior_or_prior == 'prior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False)
                elif args.dataset == 'syn_unobs':
                    if args.posterior_or_prior == 'posterior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=True)
                    elif args.posterior_or_prior == 'prior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False)
                    elif args.posterior_or_prior == 'observed_dist':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False,
                                                   observed_dist=True)
                elif args.dataset == 'project_star':
                    if args.posterior_or_prior == 'posterior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=True)
                    elif args.posterior_or_prior == 'prior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False)
                elif args.dataset == 'frugal_param':
                    if args.posterior_or_prior == 'posterior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=True)
                    elif args.posterior_or_prior == 'prior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False)
                    elif args.posterior_or_prior == 'observed_dist':
                        if 'unobs' in args.dataset_identifier:
                            dataset_info = load_smcabc(args.dataset,
                                                       DATA_SUBFOLDER,
                                                       itr,
                                                       posterior=False,
                                                       observed_dist=True,
                                                       drop_unobs_cov=True)
                        else:
                            dataset_info = load_smcabc(args.dataset,
                                                       DATA_SUBFOLDER,
                                                       itr,
                                                       posterior=False,
                                                       observed_dist=True)
                elif args.dataset == 'causl':
                    if args.posterior_or_prior == 'posterior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=True)
                    elif args.posterior_or_prior == 'prior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False)
                    elif args.posterior_or_prior == 'observed_dist':
                        if 'unobs' in args.dataset_identifier:
                            dataset_info = load_smcabc(args.dataset,
                                                       DATA_SUBFOLDER,
                                                       itr,
                                                       posterior=False,
                                                       observed_dist=True,
                                                       drop_unobs_cov=True)
                        else:
                            dataset_info = load_smcabc(args.dataset,
                                                       DATA_SUBFOLDER,
                                                       itr,
                                                       posterior=False,
                                                       observed_dist=True)
                elif args.dataset == 'e401k':
                    if args.posterior_or_prior == 'posterior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=True)
                    elif args.posterior_or_prior == 'prior':
                        dataset_info = load_smcabc(args.dataset,
                                                   DATA_SUBFOLDER,
                                                   itr,
                                                   posterior=False)
                updated_dataset_identifier = args.dataset_identifier.split('/')[0]
                print(f'Updated dataset identifier: {updated_dataset_identifier}')

            else:
                raise ValueError('Unknown or not implemented dataset!')

            estimated_ate = ate_estimators.bootstrap_ate_inference(
                outcome=dataset_info['outcome'],
                treatment=dataset_info['treatment'],
                data=dataset_info['data'],
                dataset_identifier=
                f'{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{itr}',
                set_of_estimators=ALL_ESTIMATORS,
                repeats=1)

            logger.info(f'True ATE: {dataset_info["true_ate"]}')
            # Append to the dataframe
            estimated_ate['df'] = itr
            estimated_ate['true_ate'] = dataset_info['true_ate']
            ate_df = pd.concat([ate_df, estimated_ate])
            logger.info('#' * 50)

        logger.info(f'Computed ATEs for {args.num_replications} replications.')

        # Save the dataframe to a csv file
        if args.dataset == 'lalonde':
            if args.gen_method == 'credence' or args.gen_method == 'modified_credence':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.experiment_identifier}_ate.csv'
            elif args.gen_method == 'smcabc':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.posterior_or_prior}_ate.csv'
            else:
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_ate.csv'
        elif args.dataset == 'apo_acic':
            ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_ate.csv'
        elif args.dataset == 'kunzel':
            ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_ate.csv'
        elif args.dataset == 'syn_linear':
            if args.gen_method == 'credence' or args.gen_method == 'modified_credence':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.experiment_identifier}_ate.csv'
            else:
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_ate.csv'
        elif args.dataset == 'syn_unobs':
            if args.gen_method == 'smcabc':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.posterior_or_prior}_ate.csv'
            else:
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_ate.csv'
        elif args.dataset == 'project_star':
            if args.gen_method == 'smcabc':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.posterior_or_prior}_ate.csv'
        elif args.dataset == 'frugal_param':
            if args.gen_method == 'smcabc':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.posterior_or_prior}_ate.csv'
            else:
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_ate.csv'
        elif args.dataset == 'causl':
            if args.gen_method == 'smcabc':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.posterior_or_prior}_ate.csv'
            else:
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_ate.csv'
        elif args.dataset == 'e401k':
            if args.gen_method == 'smcabc':
                ATE_DF_FILENAME = f'{ate_df_path}{args.gen_method}_{args.dataset}_{updated_dataset_identifier}_{args.posterior_or_prior}_ate.csv'
        else:
            raise ValueError('Unknown dataset!')
        logger.info(f'Saving the ATE estimates to {ATE_DF_FILENAME}')
        ate_df.to_csv(ATE_DF_FILENAME, index=False)
'''
