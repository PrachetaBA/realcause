"""Script to compute the bias squared error for the ATE estimates produced
by the posterior and the prior after using Realcause as the simulator.

This script also generates the output visualization plots for the AUC and
the bias squared error.
"""
# Import libraries
import argparse
import os
import sys
import logging
import torch
import warnings
import yaml

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
import ot
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from data_loaders import apo, lalonde, twins

# Set logger to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
ESTIMATED_ATE_PATH = 'output/ate_estimates_models'
PLOTS_PATH = 'plots/sbice_models'
# Constants that help with the plotting of the results (similar to the current paper)
# Constants to define which estimators to plot
ALL_ESTIMATORS = [
    'Diff. of Mean',
    'Linear DML',
    'Gradient Boosting Trees DML',
    'Doubly Robust (Linear)',
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
    'Diff. of Mean',
    'Linear T Learner',
    'Linear S Learner',
    'Linear X Learner',
    'Gradient Boosting Trees T Learner',
    'Gradient Boosting Trees S Learner',
    'Gradient Boosting Trees X Learner',
    'Random Forest T Learner',
    'Random Forest S Learner',
    'Random Forest X Learner',
    'Causal Forest'
]
ALL_TICKS = 16
META_TICKS = 11
ALL_SHORT_TICKLABELS = [
    'Naive',
    'Linear DML',
    'GBT DML',
    'Linear DR',
    'Linear T',
    'Linear S',
    'Linear X',
    'GBT T',
    'GBT S',
    'GBT X',
    'RF T',
    'RF S',
    'RF X',
    'Causal BART',
    'Causal Forest',
    'TMLE'
]
META_SHORT_TICKLABELS = [
    'Naive',
    'Linear T',
    'Linear S',
    'Linear X',
    'GBT T',
    'GBT S',
    'GBT X',
    'RF T',
    'RF S',
    'RF X',
    'Causal Forest'
]
# A more comprehensive set of estimators - the differences amongst which may not be minor
CLASS_ESTIMATORS = [
    'Diff. of Mean',
    'Linear X Learner',
    'Gradient Boosting Trees X Learner',
    'Linear DML',
    'Gradient Boosting Trees DML',
    'Doubly Robust (Linear)',
    'Causal BART',
    'TMLE'
]
CLASS_SHORT_TICKLABELS = [
    'Naive', 'X (Lin)', 'X (GBT)', 'DML (Lin)', 'DML (GBT)', 'DR (Lin)', 'BART', 'TMLE'
]
NUM_SAMPLES = 50


##############################################################
# Functions to extract data for the plots ####################
##############################################################
def extract_estimated_ate_base(ds_name, ds_id, sample_size, expt_id=None, realcause_only=False):
    """Extract the estimated ATE for the base datasets."""
    if realcause_only:
        ate_folder = 'output/ate_estimates'
    else:
        ate_folder = 'output/ate_estimates_models'
    if expt_id is None:
        df = pd.read_csv(
            f'{ate_folder}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_base_ate.csv',
            index_col='df')
    else:
        df = pd.read_csv(
            f'{ate_folder}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_base_ate.csv',
            index_col='df')
    # Get the true_ate
    true_ate = df['true_ate'].values[0]
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    # Return df and the true ate
    return df, true_ate


def extract_regret_base(ds_name, ds_id, sample_size, expt_id=None, realcause_only=False):
    """Extract the regret for all base datasets.

    Regret = |ATE_estimated - ATE_true| for every estimator
    """
    if realcause_only:
        ate_folder = 'output/ate_estimates'
    else:
        ate_folder = 'output/ate_estimates_models'
    if expt_id is None:
        df = pd.read_csv(
            f'{ate_folder}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_base_ate.csv',
            index_col='df')
    else:
        df = pd.read_csv(
            f'{ate_folder}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_base_ate.csv',
            index_col='df')
    df = df.subtract(df['true_ate'], axis=0)
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


def extract_estimated_ate(ds_name,
                          ds_id,
                          sample_size,
                          posterior_or_prior=None,
                          expt_id=None,
                          realcause_only=False):
    """Extract the estimated ATE for the generated datasets for a specific experimental setting
    of SMC-ABC."""
    if realcause_only:
        ate_folder = 'output/ate_estimates'
    else:
        ate_folder = 'output/ate_estimates_models'
    df = pd.read_csv(
        f'{ate_folder}/{ds_name}_{ds_id}_{sample_size}/'
        f'{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_{posterior_or_prior}_ate.csv',
        index_col='df')
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


def extract_regret(ds_name,
                   ds_id,
                   sample_size,
                   expt_id=None,
                   posterior_or_prior=None,
                   realcause_only=False):
    """Extract the regret for the generated datasets for the specific experimental setting
    of SMC-ABC."""
    if realcause_only:
        ate_folder = 'output/ate_estimates'
    else:
        ate_folder = 'output/ate_estimates_models'
    df = pd.read_csv(
        f'{ate_folder}/{ds_name}_{ds_id}_{sample_size}/'
        f'{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_{posterior_or_prior}_ate.csv',
        index_col='df')
    # Compute the regret
    df = df.subtract(df['true_ate'], axis=0)
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


##############################################################
# Functions to compute the statistics for the plots ##########
##############################################################


def rankcorr_perm_test(source_df, post_df, prior_df):
    """Computes the p-values for the Spearman rank correlation for the set of estimators
    for the posterior-source and the prior-source dataframe."""

    def post_rank_statistic(x,):
        rs = stats.spearmanr(x, post_df, nan_policy='omit').statistic
        transformed = rs * np.sqrt(len(x) - 2 / ((rs + 1.0) * (1.0 - rs)))
        return transformed

    def prior_rank_statistic(x,):
        rs = stats.spearmanr(x, prior_df, nan_policy='omit').statistic
        transformed = rs * np.sqrt(len(x) - 2 / ((rs + 1.0) * (1.0 - rs)))
        return transformed

    post_res = stats.permutation_test((source_df,),
                                      post_rank_statistic,
                                      alternative='two-sided',
                                      permutation_type='pairings')
    prior_res = stats.permutation_test((source_df,),
                                       prior_rank_statistic,
                                       alternative='two-sided',
                                       permutation_type='pairings')
    return post_res.pvalue, prior_res.pvalue


##############################################################
# Functions to plot the results ##############################
##############################################################


def plot_bias_squared_error(estimators='class',
                            ds_name=None,
                            ds_id=None,
                            sample_size=None,
                            expt_id=None,
                            distance_function=None,
                            ylims=[None, None],
                            realcause_only=False):
    """Plot the bias squared error for the generated datasets for the specific experimental setting
    of SMC-ABC."""
    """Generates boxplots of the bias squared error for estimators across different generative methods."""
    df_source = None

    # Create the combined dataframe
    df = []
    # Extract the regret for the base dataset
    df_source = extract_regret_base(ds_name, ds_id, sample_size, expt_id, realcause_only)
    # Extract the regret for the posterior dataset
    df_post = extract_regret(ds_name, ds_id, sample_size, expt_id, 'posterior', realcause_only)
    df_post = (df_post - df_source.iloc[0])**2
    df_post = df_post.stack().reset_index().rename(columns={
        'level_0': 'Identifier', 'level_1': 'Method', 0: 'ATE'
    })
    df_post['Identifier'] = r'$\text{BSE}_{\text{post}}$'
    # Extract the regret for the prior dataset
    df_prior = extract_regret(ds_name, ds_id, sample_size, expt_id, 'prior', realcause_only)
    df_prior = (df_prior - df_source.iloc[0])**2
    df_prior = df_prior.stack().reset_index().rename(columns={
        'level_0': 'Identifier', 'level_1': 'Method', 0: 'ATE'
    })
    df_prior['Identifier'] = r'$\text{BSE}_{\text{prior}}$'
    # Combine the dataframes
    df = pd.concat([df_post, df_prior], ignore_index=True)
    setting_colors = {
        r'$\text{BSE}_{\text{prior}}$': sns.color_palette('muted')[2],
        r'$\text{BSE}_{\text{post}}$': sns.color_palette('muted')[6]
    }
    # Specify the colors for the source and the posterior and the prior
    source_color = sns.color_palette('pastel')[3]
    # Depending on the identifiers present, extract the list of colors to be used
    identifiers = df['Identifier'].unique()
    colors = [setting_colors[identifier] for identifier in identifiers]

    # Create the boxplots depending on the estimators to be plotted
    plt.figure(figsize=(6, 5))
    if estimators == 'class':
        order = CLASS_ESTIMATORS
        ticks = len(CLASS_ESTIMATORS)
        short_ticklabels = CLASS_SHORT_TICKLABELS
    elif estimators == 'all':
        order = ALL_ESTIMATORS
        ticks = len(ALL_ESTIMATORS)
        short_ticklabels = ALL_SHORT_TICKLABELS

    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams.update({'legend.fontsize': font_size})
    plt.rcParams.update({'axes.labelsize': font_size})
    plt.rcParams.update({'axes.titlesize': font_size})
    plt.rcParams.update({'xtick.labelsize': font_size})
    plt.rcParams.update({'ytick.labelsize': font_size})
    ax = sns.boxplot(data=df,
                     y='ATE',
                     x='Method',
                     hue='Identifier',
                     orient='v',
                     palette=colors,
                     order=order,
                     linewidth=1.1,
                     width=0.9,
                     linecolor='black',
                     showfliers=False)

    # Put a vertical line between each x-tick label
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.grid(True, which='minor', color='black', lw=0.3)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.ylabel(f'Bias Squared Error')
    plt.xlabel('')

    if ds_name == 'lalonde' and ds_id == 'cps1':
        dataset_name = 'Lalonde (CPS)'
    elif ds_name == 'lalonde' and ds_id == 'psid1':
        dataset_name = 'Lalonde (PSID)'
    elif ds_name == 'lalonde' and ds_id == 'rct':
        dataset_name = 'Lalonde (RCT)'
    elif ds_name == 'twins':
        dataset_name = 'Twins'
    elif ds_name == 'postgres':
        dataset_name = 'Postgres'
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')

    if realcause_only:
        folder = 'plots/sbice'
    else:
        folder = 'plots/sbice_models'
    folder_path = f'{folder}/{ds_name}_{ds_id}_{sample_size}'
    os.makedirs(folder_path, exist_ok=True)
    figure_path = f'{folder_path}/bse-estimators-{estimators}-expt_{expt_id}.png'
    # figure_path = f'{folder_path}/bse-estimators-{estimators}-expt_{expt_id}.pdf'
    print(f'Saving figure to {figure_path}')

    # Add a horizontal line at 0
    plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
    if ylims[0] is not None:
        plt.ylim(ymin=ylims[0], ymax=ylims[1])
    plt.xticks(np.arange(ticks), short_ticklabels, rotation=45)
    # plt.title(f'Bias Squared Error for {dataset_name}, Experiment {expt_id}')
    plt.savefig(figure_path, bbox_inches='tight', dpi=300)


def compute_mean_bse(ds_name,
                     ds_id,
                     sample_size,
                     expt_id,
                     estimators='class',
                     realcause_only=False,
                     remove_outliers=False):
    """Compute the mean squared error between the posterior and source
    and prior-source estimated ATEs."""
    post_df = extract_regret(ds_name, ds_id, sample_size, expt_id, 'posterior', realcause_only)
    # Drop rows with NaN values
    # post_df = post_df.dropna()
    if estimators == 'class':
        post = post_df[post_df.columns.intersection(CLASS_ESTIMATORS)]
    elif estimators == 'all':
        post = post_df[post_df.columns.intersection(ALL_ESTIMATORS)]

    prior_df = extract_regret(ds_name, ds_id, sample_size, expt_id, 'prior', realcause_only)
    # Drop rows with NaN values
    # prior_df = prior_df.dropna()
    if estimators == 'class':
        prior = prior_df[prior_df.columns.intersection(CLASS_ESTIMATORS)]
    elif estimators == 'all':
        prior = prior_df[prior_df.columns.intersection(ALL_ESTIMATORS)]

    post_bse = {}
    prior_bse = {}
    if estimators == 'class':
        list_estimators = CLASS_ESTIMATORS
    elif estimators == 'all':
        list_estimators = ALL_ESTIMATORS
    # For real datasets, we do not have a distribution
    if ds_name in ['lalonde', 'postgres', 'twins']:
        source = extract_regret_base(ds_name, ds_id, sample_size, expt_id, realcause_only)
        for col in list_estimators:
            source_val = source[col].iloc[0]
            if remove_outliers:
                # Use IQR method to remove outliers (and then compute the mean)
                post_bse[col] = ((post[col] - source_val)**2).dropna()
                # Find the IQR
                post_q1 = post_bse[col].quantile(0.25)
                post_q3 = post_bse[col].quantile(0.75)
                post_iqr = post_q3 - post_q1
                # Remove outliers
                post_bse[col] = post_bse[col][(post_bse[col] > post_q1 - 1.5 * post_iqr) &
                                              (post_bse[col] < post_q3 + 1.5 * post_iqr)]
                # Compute the mean
                post_bse[col] = post_bse[col].mean()

                # Repeat for the prior
                prior_bse[col] = ((prior[col] - source_val)**2).dropna()
                prior_q1 = prior_bse[col].quantile(0.25)
                prior_q3 = prior_bse[col].quantile(0.75)
                prior_iqr = prior_q3 - prior_q1
                prior_bse[col] = prior_bse[col][(prior_bse[col] > prior_q1 - 1.5 * prior_iqr) &
                                                (prior_bse[col] < prior_q3 + 1.5 * prior_iqr)]
                # Compute the mean
                prior_bse[col] = prior_bse[col].mean()
            else:
                post_bse[col] = ((post[col] - source_val)**2).dropna().mean()
                prior_bse[col] = ((prior[col] - source_val)**2).dropna().mean()
    else:
        source = extract_regret(ds_name, ds_id, sample_size, expt_id, realcause_only)
        for col in list_estimators:
            if remove_outliers:
                # Use IQR method to remove outliers (and then compute the mean)
                post_bse[col] = ((post[col] - source[col])**2).dropna()
                # Find the IQR
                post_q1 = post_bse[col].quantile(0.25)
                post_q3 = post_bse[col].quantile(0.75)
                post_iqr = post_q3 - post_q1
                # Remove outliers
                post_bse[col] = post_bse[col][(post_bse[col] > post_q1 - 1.5 * post_iqr) &
                                              (post_bse[col] < post_q3 + 1.5 * post_iqr)]
                # Compute the mean
                post_bse[col] = post_bse[col].mean()

                # Repeat for the prior
                prior_bse[col] = ((prior[col] - source[col])**2).dropna()
                prior_q1 = prior_bse[col].quantile(0.25)
                prior_q3 = prior_bse[col].quantile(0.75)
                prior_iqr = prior_q3 - prior_q1
                prior_bse[col] = prior_bse[col][(prior_bse[col] > prior_q1 - 1.5 * prior_iqr) &
                                                (prior_bse[col] < prior_q3 + 1.5 * prior_iqr)]
                # Compute the mean
                prior_bse[col] = prior_bse[col].mean()
            else:
                post_bse[col] = ((post[col] - source[col])**2).mean()
                prior_bse[col] = ((prior[col] - source[col])**2).mean()

    # Convert this to a pandas dataframe
    bse_df = pd.DataFrame({'Prior-Source BSE': prior_bse, 'Posterior-Source BSE': post_bse})
    logger.info(f'BSE dataframe {bse_df}')
    if realcause_only:
        folder = 'plots/sbice'
    else:
        folder = 'plots/sbice_models'
    folder_path = f'{folder}/{ds_name}_{ds_id}_{sample_size}'
    os.makedirs(folder_path, exist_ok=True)
    if remove_outliers:
        bse_df.to_csv(
            f'{folder_path}/meanbse-estimators-{estimators}-expt_{expt_id}-outliers_removed.csv',
            index=True)
    else:
        bse_df.to_csv(f'{folder_path}/meanbse-estimators-{estimators}-expt_{expt_id}.csv',
                      index=True)


def compute_median_bse(ds_name,
                       ds_id,
                       sample_size,
                       expt_id,
                       estimators='class',
                       realcause_only=False):
    """Compute the median bias squared error between the posterior and source
    and prior-source estimated ATEs."""
    post_df = extract_regret(ds_name, ds_id, sample_size, expt_id, 'posterior', realcause_only)
    # Drop rows with NaN values
    # post_df = post_df.dropna()
    if estimators == 'class':
        post = post_df[post_df.columns.intersection(CLASS_ESTIMATORS)]
    elif estimators == 'all':
        post = post_df[post_df.columns.intersection(ALL_ESTIMATORS)]

    prior_df = extract_regret(ds_name, ds_id, sample_size, expt_id, 'prior', realcause_only)
    # Drop rows with NaN values
    # prior_df = prior_df.dropna()
    if estimators == 'class':
        prior = prior_df[prior_df.columns.intersection(CLASS_ESTIMATORS)]
    elif estimators == 'all':
        prior = prior_df[prior_df.columns.intersection(ALL_ESTIMATORS)]

    post_bse = {}
    prior_bse = {}
    if estimators == 'class':
        list_estimators = CLASS_ESTIMATORS
    elif estimators == 'all':
        list_estimators = ALL_ESTIMATORS
    # For real datasets, we do not have a distribution
    if ds_name in ['lalonde', 'postgres', 'twins']:
        source = extract_regret_base(ds_name, ds_id, sample_size, expt_id, realcause_only)
        for col in list_estimators:
            source_val = source[col].iloc[0]
            post_bse[col] = ((post[col] - source_val)**2).dropna().median()
            prior_bse[col] = ((prior[col] - source_val)**2).dropna().median()
    else:
        source = extract_regret(ds_name, ds_id, sample_size, expt_id, realcause_only)
        for col in list_estimators:
            post_bse[col] = ((post[col] - source[col])**2).median()
            prior_bse[col] = ((prior[col] - source[col])**2).median()

    # Convert this to a pandas dataframe
    bse_df = pd.DataFrame({'Prior-Source BSE': prior_bse, 'Posterior-Source BSE': post_bse})
    logger.info(f'Median BSE dataframe {bse_df}')
    if realcause_only:
        folder = 'plots/sbice'
    else:
        folder = 'plots/sbice_models'
    folder_path = f'{folder}/{ds_name}_{ds_id}_{sample_size}'
    os.makedirs(folder_path, exist_ok=True)
    bse_df.to_csv(f'{folder_path}/medianbse-estimators-{estimators}-expt_{expt_id}.csv', index=True)


def mmd_distance(source, generated, sigma: float = 1.0) -> float:
    """Picklable MMD distance using an RBF kernel, CPU torch only."""
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(source), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(generated), dtype=torch.float32)

        def rbf(a, b):
            diff = a.unsqueeze(1) - b.unsqueeze(0)    # [n, m, d]
            dist2 = (diff * diff).sum(dim=-1)
            return torch.exp(-dist2 / (2.0 * sigma * sigma))

        k_xx = rbf(x, x)
        k_yy = rbf(y, y)
        k_xy = rbf(x, y)
        mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        return float(mmd2)


def compute_mmd(ds_name,
                ds_id,
                sample_size,
                expt_id,
                distance_function='sliced_wass',
                smc_id=None,
                realcause_only=False):
    """Compute the MMD distance between the source and generated datasets.

    Automatically detects the prefix for sample files (rc_, ff_, or no prefix) by checking
    which files exist in the directory.
    """
    smc_path = f'data/smc_abc/{ds_name}_{ds_id}_{sample_size}_dist_{distance_function}_expt_{expt_id}/{smc_id}'
    source_df = pd.read_csv(f'{smc_path}/observed.csv')
    source_np = source_df.to_numpy()

    if ds_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif ds_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    elif ds_name == 'twins':
        outcome_col = 'yf'
        treatment_col = 't'
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')

    # Auto-detect prefix by trying different prefixes in order: 'rc_', 'ff_', then no prefix
    possible_prefixes = ['rc_', 'ff_', '']
    prefix = None
    for test_prefix in possible_prefixes:
        test_file = f'{smc_path}/{test_prefix}posterior_sample_0.csv'
        if os.path.exists(test_file):
            prefix = test_prefix
            break
    if prefix is None:
        raise FileNotFoundError(
            f'Could not find sample files in {smc_path}. '
            f'Tried: rc_posterior_sample_0.csv, ff_posterior_sample_0.csv, posterior_sample_0.csv')

    posterior_mmds = []
    prior_mmds = []
    for itr in range(NUM_SAMPLES):
        posterior_df = pd.read_csv(f'{smc_path}/{prefix}posterior_sample_{itr}.csv')
        prior_df = pd.read_csv(f'{smc_path}/{prefix}prior_sample_{itr}.csv')
        if outcome_col not in posterior_df.columns:
            posterior_df[outcome_col] = posterior_df['Y1'] * posterior_df[
                treatment_col] + posterior_df['Y0'] * (1 - posterior_df[treatment_col])
            posterior_df.drop(columns=['Y1', 'Y0'], inplace=True)
        if outcome_col not in prior_df.columns:
            prior_df[outcome_col] = prior_df['Y1'] * prior_df[treatment_col] + prior_df['Y0'] * (
                1 - prior_df[treatment_col])
            prior_df.drop(columns=['Y1', 'Y0'], inplace=True)
        post_np = posterior_df.to_numpy()
        prior_np = prior_df.to_numpy()
        # Compute the MMD distance between the post_df and source_df
        post_mmd = mmd_distance(source_np, post_np, sigma=1.0)
        prior_mmd = mmd_distance(source_np, prior_np, sigma=1.0)
        posterior_mmds.append(post_mmd)
        prior_mmds.append(prior_mmd)

    posterior_mean_mmd = np.nanmean(posterior_mmds)
    prior_mean_mmd = np.nanmean(prior_mmds)
    posterior_std_mmd = np.nanstd(posterior_mmds)
    prior_std_mmd = np.nanstd(prior_mmds)
    mmd_df = pd.DataFrame({'Posterior-Source MMD': posterior_mmds, 'Prior-Source MMD': prior_mmds})
    logger.info(f'Posterior-Source MMD: {posterior_mean_mmd:.4f} +/- {posterior_std_mmd:.4f}')
    logger.info(f'Prior-Source MMD: {prior_mean_mmd:.4f} +/- {prior_std_mmd:.4f}')
    # Save the MMD distances to a csv file
    mmd_df = pd.DataFrame({'Posterior-Source MMD': posterior_mmds, 'Prior-Source MMD': prior_mmds})
    if realcause_only:
        folder = 'plots/sbice'
    else:
        folder = 'plots/sbice_models'
    folder_path = f'{folder}/{ds_name}_{ds_id}_{sample_size}'
    os.makedirs(folder_path, exist_ok=True)
    mmd_df.to_csv(f'{folder_path}/mmd-expt_{expt_id}.csv', index=True)


def compute_sliced_wass(ds_name,
                        ds_id,
                        sample_size,
                        expt_id,
                        distance_function='sliced_wass',
                        smc_id=None,
                        realcause_only=False):
    """Compute the sliced-Wasserstein distance between the source and generated datasets.

    Automatically detects the prefix for sample files (rc_, ff_, or no prefix) by checking
    which files exist in the directory.
    """
    smc_path = f'data/smc_abc/{ds_name}_{ds_id}_{sample_size}_dist_{distance_function}_expt_{expt_id}/{smc_id}'
    source_df = pd.read_csv(f'{smc_path}/observed.csv')
    source_np = source_df.to_numpy()

    if ds_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif ds_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    elif ds_name == 'twins':
        outcome_col = 'yf'
        treatment_col = 't'
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')

    # Auto-detect prefix by trying different prefixes in order: 'rc_', 'ff_', then no prefix
    possible_prefixes = ['rc_', 'ff_', '']
    prefix = None
    for test_prefix in possible_prefixes:
        test_file = f'{smc_path}/{test_prefix}posterior_sample_0.csv'
        if os.path.exists(test_file):
            prefix = test_prefix
            break
    if prefix is None:
        raise FileNotFoundError(
            f'Could not find sample files in {smc_path}. '
            f'Tried: rc_posterior_sample_0.csv, ff_posterior_sample_0.csv, posterior_sample_0.csv')

    posterior_swds = []
    prior_swds = []
    for itr in range(NUM_SAMPLES):
        posterior_df = pd.read_csv(f'{smc_path}/{prefix}posterior_sample_{itr}.csv')
        prior_df = pd.read_csv(f'{smc_path}/{prefix}prior_sample_{itr}.csv')
        if outcome_col not in posterior_df.columns:
            posterior_df[outcome_col] = posterior_df['Y1'] * posterior_df[
                treatment_col] + posterior_df['Y0'] * (1 - posterior_df[treatment_col])
            posterior_df.drop(columns=['Y1', 'Y0'], inplace=True)
        if outcome_col not in prior_df.columns:
            prior_df[outcome_col] = prior_df['Y1'] * prior_df[treatment_col] + prior_df['Y0'] * (
                1 - prior_df[treatment_col])
            prior_df.drop(columns=['Y1', 'Y0'], inplace=True)
        post_np = posterior_df.to_numpy()
        prior_np = prior_df.to_numpy()
        # Compute the sliced wasserstein distance between the post_df and source_df
        post_swd = ot.sliced_wasserstein_distance(source_np, post_np, p=2, n_projections=100)
        prior_swd = ot.sliced_wasserstein_distance(source_np, prior_np, p=2, n_projections=100)
        posterior_swds.append(post_swd)
        prior_swds.append(prior_swd)

    posterior_mean_swd = np.nanmean(posterior_swds)
    prior_mean_swd = np.nanmean(prior_swds)
    posterior_std_swd = np.nanstd(posterior_swds)
    prior_std_swd = np.nanstd(prior_swds)
    swd_df = pd.DataFrame({'Posterior-Source SWD': posterior_swds, 'Prior-Source SWD': prior_swds})
    logger.info(f'Posterior-Source SWD: {posterior_mean_swd:.2f} +/- {posterior_std_swd:.2f}')
    logger.info(f'Prior-Source SWD: {prior_mean_swd:.2f} +/- {prior_std_swd:.2f}')
    # Save the Sliced Wasserstein Distances to a csv file
    swds_df = pd.DataFrame({'Posterior-Source SWD': posterior_swds, 'Prior-Source SWD': prior_swds})
    if realcause_only:
        folder = 'plots/sbice'
    else:
        folder = 'plots/sbice_models'
    folder_path = f'{folder}/{ds_name}_{ds_id}_{sample_size}'
    os.makedirs(folder_path, exist_ok=True)
    swds_df.to_csv(f'{folder_path}/swd-expt_{expt_id}.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_config', type=str, default=None)
    parser.add_argument('--experiment_number', type=int, default=None)
    parser.add_argument('--estimators', type=str, default='all', choices=['class', 'all'])
    parser.add_argument('--ylims', type=float, nargs=2, required=False, default=[None, None])
    parser.add_argument('--smc_id', type=int, default=None)
    parser.add_argument('--exec_function',
                        type=str,
                        default='all',
                        choices=[
                            'all',
                            'plot_bias_squared_error',
                            'compute_mean_bse',
                            'compute_sliced_wass',
                            'compute_median_bse',
                            'compute_mmd'
                        ])
    parser.add_argument('--remove_outliers', action='store_true')
    args = parser.parse_args()

    with open(args.experiment_config, 'r', encoding='utf-8') as file:
        expt_configs = yaml.safe_load(file)
    expt_config = expt_configs[f'experiment_{args.experiment_number}']
    dataset_name = expt_config['dataset_name']
    dataset_identifier = expt_config['dataset_identifier']
    sample_size = str(expt_config['sample_size'])
    expt_id = args.experiment_number
    distance_function = expt_config['distance']
    transformation = expt_config['transform']

    if dataset_name == 'twins':
        realcause_only = True
    else:
        realcause_only = False
    if args.exec_function == 'all' or args.exec_function == 'plot_bias_squared_error':
        plot_bias_squared_error(estimators=args.estimators,
                                ds_name=dataset_name,
                                ds_id=dataset_identifier,
                                sample_size=sample_size,
                                expt_id=expt_id,
                                distance_function=distance_function,
                                ylims=args.ylims,
                                realcause_only=realcause_only)
    if args.exec_function == 'all' or args.exec_function == 'compute_mean_bse':
        compute_mean_bse(ds_name=dataset_name,
                         ds_id=dataset_identifier,
                         sample_size=sample_size,
                         expt_id=expt_id,
                         estimators=args.estimators,
                         realcause_only=realcause_only,
                         remove_outliers=args.remove_outliers)
    if args.exec_function == 'all' or args.exec_function == 'compute_sliced_wass':
        compute_sliced_wass(ds_name=dataset_name,
                            ds_id=dataset_identifier,
                            sample_size=sample_size,
                            expt_id=expt_id,
                            distance_function=distance_function,
                            smc_id=args.smc_id,
                            realcause_only=realcause_only)
    if args.exec_function == 'all' or args.exec_function == 'compute_median_bse':
        compute_median_bse(ds_name=dataset_name,
                           ds_id=dataset_identifier,
                           sample_size=sample_size,
                           expt_id=expt_id,
                           estimators=args.estimators,
                           realcause_only=realcause_only)
    if args.exec_function == 'all' or args.exec_function == 'compute_mmd':
        compute_mmd(ds_name=dataset_name,
                    ds_id=dataset_identifier,
                    sample_size=sample_size,
                    expt_id=expt_id,
                    distance_function=distance_function,
                    smc_id=args.smc_id,
                    realcause_only=realcause_only)
