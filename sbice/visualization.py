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
ESTIMATED_ATE_PATH = 'output/ate_estimates'
PLOTS_PATH = 'plots/sbice'
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
        df = pd.read_csv(f'output/ate_estimates/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_base_ate.csv',
                         index_col='df')
    else:
        df = pd.read_csv(f'output/ate_estimates/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_base_ate.csv',
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
        df = pd.read_csv(f'output/ate_estimates/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_base_ate.csv',
                         index_col='df')
    else:
        df = pd.read_csv(f'output/ate_estimates/{ds_name}_{ds_id}_{sample_size}/{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_base_ate.csv',
                         index_col='df')
    df = df.subtract(df['true_ate'], axis=0)
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df


def extract_estimated_ate(ds_name, ds_id, sample_size, posterior_or_prior=None, expt_id=None):
    """Extract the estimated ATE for the generated datasets for a specific experimental setting
    of SMC-ABC."""
    df = pd.read_csv(f'output/ate_estimates/{ds_name}_{ds_id}_{sample_size}/'
                     f'{ds_name}_{ds_id}_{sample_size}_expt_{expt_id}_{posterior_or_prior}_ate.csv',
                     index_col='df')
    # Drop the true_ate column
    df = df.drop(columns=['true_ate'])
    return df

def extract_regret(ds_name, ds_id, sample_size, expt_id=None, posterior_or_prior=None):
    """Extract the regret for the generated datasets for the specific experimental setting
    of SMC-ABC."""
    df = pd.read_csv(f'output/ate_estimates/{ds_name}_{ds_id}_{sample_size}/'
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
                            ylims=[None,None]):
    """Plot the bias squared error for the generated datasets for the specific experimental setting
    of SMC-ABC."""
    """Generates boxplots of the bias squared error for estimators across different generative methods."""
    df_source = None

    # Create the combined dataframe
    df = []
    # Extract the regret for the base dataset
    df_source = extract_regret_base(ds_name, ds_id, sample_size, expt_id)
    # Extract the regret for the posterior dataset
    df_post = extract_regret(ds_name, ds_id, sample_size, expt_id, 'posterior')
    df_post = (df_post - df_source.iloc[0])**2 
    df_post = df_post.stack().reset_index().rename(columns={
        'level_0': 'Identifier', 'level_1': 'Method', 0: 'ATE'
    })
    df_post['Identifier'] = r'$\text{BSE}_{\text{post}}$'
    # Extract the regret for the prior dataset
    df_prior = extract_regret(ds_name, ds_id, sample_size, expt_id, 'prior') 
    df_prior = (df_prior - df_source.iloc[0])**2 
    df_prior = df_prior.stack().reset_index().rename(columns={
        'level_0': 'Identifier', 'level_1': 'Method', 0: 'ATE'
    })
    df_prior['Identifier'] = r'$\text{BSE}_{\text{prior}}$'
    # Combine the dataframes
    df = pd.concat([df_post, df_prior], ignore_index=True)
    print(df.head(25))
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
    elif ds_name == 'twins':
        dataset_name = 'Twins'
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')

    folder_path = f'plots/sbice/{ds_name}_{ds_id}_{sample_size}'
    os.makedirs(folder_path, exist_ok=True)
    figure_path = f'{folder_path}/bse-estimators-{estimators}-expt_{expt_id}.png'
    print(f'Saving figure to {figure_path}')

    # Add a horizontal line at 0
    plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.2)
    if ylims[0] is not None:
        plt.ylim(ymin=ylims[0], ymax=ylims[1])
    plt.xticks(np.arange(ticks), short_ticklabels, rotation=45)
    plt.title(f'Bias Squared Error for {dataset_name}, Experiment {expt_id}')
    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimators', type=str, default='all', choices=['class', 'all'])
    parser.add_argument('--ds_name', type=str, default=None)
    parser.add_argument('--ds_id', type=str, default=None)
    parser.add_argument('--sample_size', type=str, default=None)
    parser.add_argument('--expt_id', type=int, default=None)
    parser.add_argument('--distance_function', type=str, default='sliced_wass')
    parser.add_argument('--ylims', type=list, default=[None, None])
    args = parser.parse_args()
    
    plot_bias_squared_error(estimators=args.estimators,
                        ds_name=args.ds_name,
                        ds_id=args.ds_id,
                        sample_size=args.sample_size,
                        expt_id=args.expt_id,
                        distance_function=args.distance_function,
                        ylims=args.ylims)
