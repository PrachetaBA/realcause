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
import warnings

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
ESTIMATED_ATE_PATH = 'output/gen_methods_ate_estimates'
PLOTS_PATH = 'plots/gen_methods'
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


##############################################################
# Functions to extract data for the plots ####################
##############################################################
def extract_estimated_ate_base(ds_name, ds_id, sample_size, expt_id=None):
    """Extract the estimated ATE for the base datasets."""
    if expt_id is None:
        df = pd.read_csv(
            f'{ESTIMATED_ATE_PATH}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_source_ate.csv',
            index_col='df')
    else:
        df = pd.read_csv(
            f'{ESTIMATED_ATE_PATH}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_source_ate.csv',
            index_col='df')
    # Get the true_ate
    true_ate = df['true_ate'].values[0]
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    # Return df and the true ate
    return df, true_ate


def extract_regret_base(ds_name, ds_id, sample_size, expt_id=None):
    """Extract the regret for all base datasets.

    Regret = |ATE_estimated - ATE_true| for every estimator
    """
    if expt_id is None:
        df = pd.read_csv(
            f'{ESTIMATED_ATE_PATH}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_source_ate.csv',
            index_col='df')
    else:
        df = pd.read_csv(
            f'{ESTIMATED_ATE_PATH}/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_source_ate.csv',
            index_col='df')
    df = df.subtract(df['true_ate'], axis=0)
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


def extract_estimated_ate(ds_name, ds_id, sample_size, gen_method=None, expt_id=None):
    """Extract the estimated ATE for the generated datasets for a specific experimental setting
    of SMC-ABC."""
    df = pd.read_csv(
        f'{ESTIMATED_ATE_PATH}/{ds_name}_{ds_id}_{sample_size}/'
        f'{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_{gen_method}_ate.csv',
        index_col='df')
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


def extract_regret(ds_name, ds_id, sample_size, expt_id=None, gen_method=None):
    """Extract the regret for the generated datasets for the specific experimental setting
    of SMC-ABC."""
    df = pd.read_csv(
        f'{ESTIMATED_ATE_PATH}/{ds_name}_{ds_id}_{sample_size}/'
        f'{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_{gen_method}_ate.csv',
        index_col='df')
    # Compute the regret
    df = df.subtract(df['true_ate'], axis=0)
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


##############################################################
# Functions to compute the statistics for the plots ##########
##############################################################


def rankcorr_perm_test(source_df, gen_df):
    """Computes the p-values for the Spearman rank correlation for the set of estimators
    for the generated datasets from each method and the source dataframe."""

    def gendata_rank_statistic(x,):
        rs = stats.spearmanr(x, gen_df, nan_policy='omit').statistic
        transformed = rs * np.sqrt(len(x) - 2 / ((rs + 1.0) * (1.0 - rs)))
        return transformed

    gen_res = stats.permutation_test((source_df,),
                                     gendata_rank_statistic,
                                     alternative='two-sided',
                                     permutation_type='pairings')
    return gen_res.pvalue


#####################################################################
# Functions to plot the bias of estimators for each generative method
#####################################################################


def plot_bias_estimators(ds_name,
                         ds_id,
                         sample_size,
                         expt_id=None,
                         gen_method=None,
                         estimators='all',
                         ylims=[None, None],
                         regret=True):
    """Plots the bias of the estimators for the generated datasets in two ways:
    1. expt_id: True (or specified)
       Plots the bias of the estimators across all the generative methods for that specific expt_id
    2. gen_method: True (or specified)
       Plots the bias of the estimators for the specific generative method across all three expt_ids (0001, 0002, 0003)
    """
    if expt_id is not None:
        # Create the combined dataframe
        df = []
        if regret:
            df_source = extract_regret_base(ds_name, ds_id, sample_size, expt_id)
        else:
            df_source, true_ate = extract_estimated_ate_base(ds_name, ds_id, sample_size, expt_id)
        df_source = df_source.stack().reset_index().rename(columns={
            'level_0': 'Identifier', 'level_1': 'Method', 0: 'ATE'
        })
        df_source['Identifier'] = 'Source'
        gen_methods = ['credence', 'mcredence', 'realcause', 'frugalflows']
        for gen_method in gen_methods:
            if gen_method != 'source':
                if regret:
                    df_gm = extract_regret(ds_name, ds_id, sample_size, expt_id, gen_method)
                else:
                    df_gm = extract_estimated_ate(ds_name, ds_id, sample_size, gen_method, expt_id)
                df_gm = df_gm.stack().reset_index().rename(columns={
                    'level_0': 'Identifier', 'level_1': 'Method', 0: 'ATE'
                })
                if gen_method == 'credence':
                    df_gm['Identifier'] = 'Credence'
                elif gen_method == 'mcredence':
                    df_gm['Identifier'] = 'modified-Credence'
                elif gen_method == 'realcause':
                    df_gm['Identifier'] = 'RealCause'
                elif gen_method == 'frugalflows':
                    df_gm['Identifier'] = 'FrugalFlows'
                df.append(df_gm)
        df = pd.concat(df, ignore_index=True)

        # Create specific color palette for the generative methods
        if 'mcredence' in gen_methods:
            method_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        else:
            method_colors = ['#66c2a5', '#8da0cb', '#e78ac3']
        source_color = '#e5c494'

        # Create the boxplots depending on the estimators to be plotted
        plt.figure(figsize=(8, 8))
        if estimators == 'all':
            order = ALL_ESTIMATORS
            ticks = ALL_TICKS
            short_ticklabels = ALL_SHORT_TICKLABELS
        elif estimators == 'meta':
            order = META_ESTIMATORS
            ticks = META_TICKS
            short_ticklabels = META_SHORT_TICKLABELS
        elif estimators == 'class':
            order = CLASS_ESTIMATORS
            ticks = len(CLASS_ESTIMATORS)
            short_ticklabels = CLASS_SHORT_TICKLABELS
        ax = sns.boxenplot(data=df,
                           y='ATE',
                           x='Method',
                           hue='Identifier',
                           orient='v',
                           palette=method_colors,
                           order=order,
                           showfliers=False)
        if df_source is not None:
            sns.stripplot(data=df_source,
                          y='ATE',
                          x='Method',
                          orient='v',
                          color=source_color,
                          order=order,
                          size=4,
                          marker='D',
                          jitter=True,
                          linewidth=0.8,
                          ax=ax)

        # Put a vertical line between each x-tick label
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.grid(True, which='minor', color='black', lw=0.3)
        # Extract the handles
        handles, labels = ax.get_legend_handles_labels()

        if df_source is not None:
            source_diamond = mlines.Line2D([], [],
                                           color=source_color,
                                           marker='D',
                                           linestyle='None',
                                           markersize=8,
                                           label='Source')
            handles.append(source_diamond)
            labels.append('Source')

        # Place the legend below the x-axis
        ncols = 3
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=ncols)
        plt.xlabel('Causal Estimator')
        plt.ylabel(f'Bias')

        if ds_name == 'lalonde':
            if ds_id == 'psid1':
                dataset_name = 'Lalonde (PSID)'
            elif ds_id == 'cps1':
                dataset_name = 'Lalonde (CPS)'
            else:
                raise ValueError(f'Dataset identifier {ds_id} not implemented')

        if expt_id == '0001':
            setting_name = 'Learned ATE'
        elif expt_id == '0002':
            setting_name = 'True ATE'
        elif expt_id == '0003':
            setting_name = 'Incorrect ATE'
        title = f'Dataset: {dataset_name} \n {setting_name}'
        plt.title(title)

        folder_path = f'{PLOTS_PATH}/{ds_name}_{ds_id}_{sample_size}'
        os.makedirs(folder_path, exist_ok=True)
        if regret:
            figure_path = f'{folder_path}/bias-estimators-{estimators}-{expt_id}-bias.png'
        else:
            figure_path = f'{folder_path}/bias-estimators-{estimators}-{expt_id}-ate.png'
        print(f'Saving figure to {figure_path}')

        # Add a horizontal line at 0
        plt.axhline(y=0.0, color='red', linestyle='--', linewidth=0.5)
        if ylims[0] and ylims[1]:
            plt.ylim(ymin=ylims[0], ymax=ylims[1])
        plt.xticks(np.arange(ticks), short_ticklabels, rotation=45)
        plt.savefig(figure_path, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimators', type=str, default='all', choices=['class', 'all'])
    parser.add_argument('--ds_name', type=str, default=None)
    parser.add_argument('--ds_id', type=str, default=None)
    parser.add_argument('--sample_size', type=str, default=None)
    parser.add_argument('--expt_id', type=str, default=None)
    parser.add_argument('--gen_method', type=str, default=None)
    parser.add_argument('--ylims', type=float, nargs=2, required=False, default=[None, None])
    args = parser.parse_args()

    plot_bias_estimators(ds_name=args.ds_name,
                         ds_id=args.ds_id,
                         sample_size=args.sample_size,
                         expt_id=args.expt_id,
                         gen_method=args.gen_method,
                         estimators=args.estimators,
                         ylims=args.ylims,
                         regret=True)
    plot_bias_estimators(ds_name=args.ds_name,
                         ds_id=args.ds_id,
                         sample_size=args.sample_size,
                         expt_id=args.expt_id,
                         gen_method=args.gen_method,
                         estimators=args.estimators,
                         ylims=args.ylims,
                         regret=False)
