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
from loading import load_gen

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
    'DML (Lin)',
    'DML (GBT)',
    'DR (Lin)',
    'T (Lin)',
    'S (Lin)',
    'X (Lin)',
    'T (GBT)',
    'S (GBT)',
    'X (GBT)',
    'T (RF)',
    'S (RF)',
    'X (RF)',
    'BART',
    'CF',
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
        source_color = 'gold'    # '#e5c494' # gold

        # Create the boxplots depending on the estimators to be plotted
        plt.figure(figsize=(9, 9))
        font_size = 22
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams.update({'legend.fontsize': font_size - 2})
        plt.rcParams.update({'axes.labelsize': font_size})
        plt.rcParams.update({'axes.titlesize': font_size})
        plt.rcParams.update({'xtick.labelsize': font_size - 6})
        plt.rcParams.update({'ytick.labelsize': font_size})
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
        palette = sns.color_palette('pastel')
        new_palette = palette[1:]

        ax = sns.boxplot(data=df,
                         y='ATE',
                         x='Method',
                         hue='Identifier',
                         orient='v',
                         palette=new_palette,
                         order=order,
                         showfliers=False,
                         linewidth=1.1,
                         width=0.9,
                         linecolor='black')
        if df_source is not None:
            sns.stripplot(data=df_source,
                          y='ATE',
                          x='Method',
                          orient='v',
                          color='gold',
                          order=order,
                          size=8,
                          marker='D',
                          jitter=True,
                          linewidth=0.8,
                          ax=ax,
                          edgecolor='black')

        # Put a vertical line between each x-tick label
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.grid(True, which='minor', color='black', lw=1.0)
        # Extract the handles
        handles, labels = ax.get_legend_handles_labels()

        if df_source is not None:
            source_diamond = mlines.Line2D([], [],
                                           color='gold',
                                           marker='D',
                                           linestyle='None',
                                           markersize=8,
                                           label='Source',
                                           markeredgecolor='black')
            handles.append(source_diamond)
            labels.append('Source')

        # Place the legend below the x-axis
        ncols = 3
        plt.legend(handles=handles,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   ncol=ncols,
                   fontsize=font_size - 4)
        plt.xlabel('')
        plt.ylabel(rf'Bias (Estimated ATE $-$ True ATE)')

        if ds_name == 'lalonde':
            if ds_id == 'psid1':
                dataset_name = 'Lalonde (PSID)'
                dataset_plot_name = 'lalonde-psid'
            elif ds_id == 'cps1':
                dataset_name = 'Lalonde (CPS)'
                dataset_plot_name = 'lalonde-cps'
            else:
                raise ValueError(f'Dataset identifier {ds_id} not implemented')
        elif ds_name == 'postgres':
            if ds_id == 'linear':
                dataset_name = 'Postgres (Linear)'
                dataset_plot_name = 'postgres-linear'
            else:
                raise ValueError(f'Dataset identifier {ds_id} not implemented')
        else:
            raise ValueError(f'Dataset name {ds_name} not implemented')

        if expt_id == '0001':
            setting_name = 'Learned ATE'
            setting_plot_name = 'flexible'
        elif expt_id == '0002':
            setting_name = 'True ATE'
            setting_plot_name = 'true'
        elif expt_id == '0003':
            setting_name = 'Incorrect ATE'
            setting_plot_name = 'incorrect'
        elif expt_id == '0004':
            setting_name = 'Flexible ATE'
            setting_plot_name = 'flexible'
        title = f'Dataset: {dataset_name} \n {setting_name}'
        # plt.title(title)

        folder_path = f'{PLOTS_PATH}/{ds_name}_{ds_id}_{sample_size}'
        os.makedirs(folder_path, exist_ok=True)
        if regret:
            figure_path = f'{folder_path}/{dataset_plot_name}-{setting_plot_name}-{estimators}-bias.png'
        else:
            figure_path = f'{folder_path}/{dataset_plot_name}-{setting_plot_name}-{estimators}-ate.png'
        print(f'Saving figure to {figure_path}')

        # Add a horizontal line at 0
        plt.axhline(y=0.0, color='red', linestyle='--', linewidth=0.8)
        if ylims[0] and ylims[1]:
            plt.ylim(ymin=ylims[0], ymax=ylims[1])
        plt.xticks(np.arange(ticks), short_ticklabels, rotation=45)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)


###############################################################################
# Functions to plot the distribution of the outcomes for the generated datasets
###############################################################################


def load_credence_generated(ds_name, ds_id, sample_size, expt_id, dataset_num):
    credence_df = pd.read_csv(
        f'data/generated_datasets/credence/expt_{expt_id}/dataset_{dataset_num}_prime.csv')
    if ds_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif ds_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    credence_df['method'] = 'Credence'

    # After this block, the columns 'outcome', 'counterfactual_outcome', and 'treatment' will be present in the dataframe
    if ds_name == 'lalonde':
        credence_df['outcome'] = credence_df['treat'] * credence_df['Y1'] + (
            1 - credence_df['treat']) * credence_df['Y0']
        credence_df['counterfactual_outcome'] = credence_df['treat'] * credence_df['Yprime0'] + (
            1 - credence_df['treat']) * credence_df['Yprime1']
        # Rename t to treatment
        credence_df.rename(columns={'treat': 'treatment'}, inplace=True)
    elif ds_name == 'postgres':
        credence_df['outcome'] = credence_df[treatment_col] * credence_df['Y1'] + (
            1 - credence_df[treatment_col]) * credence_df['Y0']
        credence_df['counterfactual_outcome'] = credence_df[treatment_col] * credence_df[
            'Yprime0'] + (1 - credence_df[treatment_col]) * credence_df['Yprime1']
        # Rename t to treatment
        credence_df.rename(columns={treatment_col: 'treatment'}, inplace=True)
    # Compute the ITE for each unit. The ITE is the Y1 - Yprime1 if treatment is 1, and Y0 - Yprime0 if treatment is 0
    credence_df['ite'] = credence_df['Y1'] - credence_df['Y0']
    credence_df['ate'] = credence_df['ite'].mean()
    # Compute the propensity score, which is the probability of treatment == 1
    credence_df['p_t'] = credence_df['treatment'].mean()
    credence_df['selection_bias'] = credence_df['treatment'] * (
        credence_df['Y1'] - credence_df['Yprime1']) + (1 - credence_df['treatment']) * (
            credence_df['Yprime0'] - credence_df['Y0'])
    # Drop the unnecessary columns
    credence_df.drop(columns=['Y1', 'Y0', 'Yprime1', 'Yprime0'], inplace=True)
    # Add in the dataset number
    credence_df['dataset_num'] = dataset_num
    # Reorder the columns
    if ds_name == 'lalonde':
        if ds_id == 'psid1':
            if expt_id == '0001':
                setting = 'flexible_ate'
            elif expt_id == '0002':
                setting = 'true_ate'
            elif expt_id == '0003':
                setting = 'incorrect_ate'
            credence_df['setting'] = setting
            credence_df = credence_df[[
                'method',
                'setting',
                'dataset_num',
                'black',
                'hispanic',
                'married',
                'nodegree',
                'age',
                'education',
                're74',
                're75',
                'treatment',
                'outcome',
                'ate',
                'p_t',
                'counterfactual_outcome',
                'ite',
                'selection_bias'
            ]]
    elif ds_name == 'postgres':
        if ds_id == 'linear':
            if expt_id == '0004':
                setting = 'flexible_ate'
            credence_df['setting'] = setting
            credence_df = credence_df[[
                'method',
                'setting',
                'dataset_num',
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                'treatment',
                'outcome',
                'ate',
                'p_t',
                'counterfactual_outcome',
                'ite',
                'selection_bias'
            ]]
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')
    return credence_df


def load_mcredence_generated(ds_name, ds_id, sample_size, expt_id, dataset_num):
    mcredence_df = pd.read_csv(
        f'data/generated_datasets/modified_credence/expt_{expt_id}/dataset_{dataset_num}_prime.csv')
    mcredence_df['method'] = 'modCredence'
    if ds_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif ds_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    mcredence_df['outcome'] = mcredence_df[treatment_col] * mcredence_df['Y1'] + (
        1 - mcredence_df[treatment_col]) * mcredence_df['Y0']
    mcredence_df['counterfactual_outcome'] = mcredence_df[treatment_col] * mcredence_df[
        'Yprime0'] + (1 - mcredence_df[treatment_col]) * mcredence_df['Yprime1']
    # Rename t to treatment
    mcredence_df.rename(columns={treatment_col: 'treatment'}, inplace=True)
    # Compute the ITE for each unit. The ITE is the Y1 - Yprime1 if treatment is 1, and Y0 - Yprime0 if treatment is 0
    mcredence_df['ite'] = mcredence_df['Y1'] - mcredence_df['Y0']
    mcredence_df['ate'] = mcredence_df['ite'].mean()
    # Compute the propensity score, which is the probability of treatment == 1
    mcredence_df['p_t'] = mcredence_df['treatment'].mean()
    mcredence_df['selection_bias'] = mcredence_df['treatment'] * (
        mcredence_df['Y1'] - mcredence_df['Yprime1']) + (1 - mcredence_df['treatment']) * (
            mcredence_df['Yprime0'] - mcredence_df['Y0'])
    # Drop the unnecessary columns
    mcredence_df.drop(columns=['Y1', 'Y0', 'Yprime1', 'Yprime0', 'Y_cf'], inplace=True)
    # Add in the dataset number
    mcredence_df['dataset_num'] = dataset_num
    # Reorder the columns
    if ds_name == 'lalonde':
        if ds_id == 'psid1':
            if expt_id == '0001':
                setting = 'flexible_ate'
            elif expt_id == '0002':
                setting = 'true_ate'
            elif expt_id == '0003':
                setting = 'incorrect_ate'
            mcredence_df['setting'] = setting
            mcredence_df = mcredence_df[[
                'method',
                'setting',
                'dataset_num',
                'black',
                'hispanic',
                'married',
                'nodegree',
                'age',
                'education',
                're74',
                're75',
                'treatment',
                'outcome',
                'ate',
                'p_t',
                'counterfactual_outcome',
                'ite',
                'selection_bias'
            ]]
    elif ds_name == 'postgres':
        if ds_id == 'linear':
            if expt_id == '0004':
                setting = 'flexible_ate'
            mcredence_df['setting'] = setting
            mcredence_df = mcredence_df[[
                'method',
                'setting',
                'dataset_num',
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                'treatment',
                'outcome',
                'ate',
                'p_t',
                'counterfactual_outcome',
                'ite',
                'selection_bias'
            ]]
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')
    return mcredence_df


def load_realcause_generated(ds_name, ds_id, sample_size, expt_id, dataset_num):
    realcause_df = pd.read_csv(
        f'data/generated_datasets/realcause/expt_{expt_id}/dataset_{dataset_num}.csv')
    realcause_df['method'] = 'Realcause'
    if ds_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif ds_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    realcause_df['ite'] = realcause_df['y1'] - realcause_df['y0']
    realcause_df['ate'] = realcause_df['ite'].mean()
    # Rename treatment_col to treatment
    realcause_df.rename(columns={treatment_col: 'treatment'}, inplace=True)
    # Compute the propensity score, which is the probability of treatment == 1
    realcause_df['p_t'] = realcause_df['treatment'].mean()
    # Drop the unnecessary columns
    realcause_df.drop(columns=['y1', 'y0'], inplace=True)
    # Add in the dataset number
    realcause_df['dataset_num'] = dataset_num
    # Reorder the columns
    if ds_name == 'lalonde':
        if ds_id == 'psid1':
            # Rename the 're78' column to 'outcome'
            realcause_df.rename(columns={'re78': 'outcome'}, inplace=True)
            if expt_id == '0001':
                setting = 'flexible_ate'
            elif expt_id == '0002':
                setting = 'true_ate'
            elif expt_id == '0003':
                setting = 'incorrect_ate'
            realcause_df['setting'] = setting
            realcause_df = realcause_df[[
                'method',
                'setting',
                'dataset_num',
                'black',
                'hispanic',
                'married',
                'nodegree',
                'age',
                'education',
                're74',
                're75',
                'treatment',
                'outcome',
                'ate',
                'p_t',
                'ite'
            ]]
    elif ds_name == 'postgres':
        if ds_id == 'linear':
            if expt_id == '0004':
                setting = 'flexible_ate'
            # Rename the 'runtime' column to 'outcome'
            realcause_df.rename(columns={outcome_col: 'outcome'}, inplace=True)
            realcause_df['setting'] = setting
            realcause_df = realcause_df[[
                'method',
                'setting',
                'dataset_num',
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                'treatment',
                'outcome',
                'ate',
                'p_t',
                'ite'
            ]]
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')
    return realcause_df


def load_frugalflows_generated(ds_name, ds_id, sample_size, expt_id, dataset_num):
    frugalflows_df = pd.read_csv(
        f'data/generated_datasets/frugalflows/expt_{expt_id}/dataset_{dataset_num}.csv')
    frugalflows_df['method'] = 'Frugalflows'
    if ds_name == 'lalonde':
        outcome_col = 're78'
        treatment_col = 'treat'
    elif ds_name == 'postgres':
        outcome_col = 'runtime'
        treatment_col = 'index_level'
    # Rename treatment_col to treatment
    frugalflows_df.rename(columns={
        treatment_col: 'treatment', outcome_col: 'outcome'
    },
                          inplace=True)
    # Compute the propensity score, which is the probability of treatment == 1
    frugalflows_df['p_t'] = frugalflows_df['treatment'].mean()
    # Add in the dataset number
    frugalflows_df['dataset_num'] = dataset_num
    # Reorder the columns
    if ds_name == 'lalonde':
        if ds_id == 'psid1':
            if expt_id == '0001':
                setting = 'flexible_ate'
            elif expt_id == '0002':
                setting = 'true_ate'
            elif expt_id == '0003':
                setting = 'incorrect_ate'
            frugalflows_df['setting'] = setting
            frugalflows_df = frugalflows_df[[
                'method',
                'setting',
                'dataset_num',
                'black',
                'hispanic',
                'married',
                'nodegree',
                'age',
                'education',
                're74',
                're75',
                'treatment',
                'outcome',
                'p_t'
            ]]
    elif ds_name == 'postgres':
        if ds_id == 'linear':
            if expt_id == '0004':
                setting = 'flexible_ate'
            frugalflows_df['setting'] = setting
            frugalflows_df = frugalflows_df[[
                'method',
                'setting',
                'dataset_num',
                'rows',
                'creation_year',
                'num_ref_tables',
                'num_joins',
                'num_group_by',
                'queries_by_user',
                'length_chars',
                'total_ref_rows',
                'treatment',
                'outcome',
                'p_t'
            ]]
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')
    return frugalflows_df


def load_source_df(ds_name, ds_id, sample_size, rc_model_path):
    if ds_name == 'lalonde' and ds_id == 'psid1':
        d = lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
        d.drop(columns=['data_id'], inplace=True)
        # Rename treatment to treatment
        d.rename(columns={'treat': 'treatment', 're78': 'outcome'}, inplace=True)
        outcome_col = 'outcome'
        categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
        continuous_vars = ['age', 'education', 're75', 're74']
        # Sort the covariates columns to put the continous first, then the categorical
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
    elif ds_name == 'postgres':
        d, d_info = apo.get_apo_data(identifier='postgres', confound_func=dataset_identifier, data_format='pandas', return_ites=False, ret_counterfactual_outcomes=False, sample_size=sample_size)
        true_ate = d_info['true_ate']
        treatment_col = d_info['treatment_col']
        outcome_col = d_info['outcome_col']
        # Concatenate the covariates, treatment and outcome columns
        d = pd.concat([d['w'], d['t'], d['y']], axis=1)
        # Rename treatment_col to treatment and outcome_col to outcome
        d.rename(columns={treatment_col: 'treatment', outcome_col: 'outcome'}, inplace=True)
    else:
        raise ValueError(f'Dataset {ds_name} not implemented')
    d['setting'] = 'source'
    d['method'] = 'Source'
    return d


def plot_outcome_distribution(ds_name, ds_id, sample_size):
    """Original function: plots all three settings (flexible, true, incorrect) with overlaid methods."""
    # Create a dataframe of all the settings and methods together
    # Pick a random dataset number
    i = np.random.randint(0, 49)
    print(f'Using dataset number {i}')

    if ds_name == 'lalonde' and ds_id == 'psid1':
        source_df = load_source_df(ds_name,
                                   ds_id,
                                   sample_size,
                                   'results/GenModelCkpts/lalonde/psid1/save')
        # Extract each of the datasets across all methods and all settings
        cred_flexible = load_credence_generated('lalonde', 'psid1', None, '0001', i)
        cred_true = load_credence_generated('lalonde', 'psid1', None, '0002', i)
        cred_incorrect = load_credence_generated('lalonde', 'psid1', None, '0003', i)
        mcred_flexible = load_mcredence_generated('lalonde', 'psid1', None, '0001', i)
        mcred_true = load_mcredence_generated('lalonde', 'psid1', None, '0002', i)
        mcred_incorrect = load_mcredence_generated('lalonde', 'psid1', None, '0003', i)
        rc_flexible = load_realcause_generated('lalonde', 'psid1', None, '0001', i)
        rc_true = load_realcause_generated('lalonde', 'psid1', None, '0002', i)
        rc_incorrect = load_realcause_generated('lalonde', 'psid1', None, '0003', i)
        ff_flexible = load_frugalflows_generated('lalonde', 'psid1', None, '0001', i)
        ff_true = load_frugalflows_generated('lalonde', 'psid1', None, '0002', i)
        ff_incorrect = load_frugalflows_generated('lalonde', 'psid1', None, '0003', i)
    elif ds_name == 'postgres':
        pass    # TODO: Implement this
    # Concatenate all the dataframes together
    df = pd.concat([
        source_df,
        cred_flexible,
        cred_true,
        cred_incorrect,
        mcred_flexible,
        mcred_true,
        mcred_incorrect,
        ff_flexible,
        ff_true,
        ff_incorrect,
        rc_flexible,
        rc_true,
        rc_incorrect
    ],
                   axis=0)
    df.reset_index(drop=True, inplace=True)

    # First subplot extract only those rows where setting isin 'flexible_ate', 'source'
    df_flex = df[df['setting'].isin(['flexible_ate', 'source'])]
    # Second subplot extract only those rows where setting isin 'true_ate', 'source'
    df_fixed = df[df['setting'].isin(['true_ate', 'source'])]
    # Third subplot extract only those rows where setting isin 'incorrect_ate', 'source'
    df_incorrect = df[df['setting'].isin(['incorrect_ate', 'source'])]

    print(df_flex.columns)

    # Ensure that there are no nans in the outcome column
    print(df_flex['outcome'].isna().sum())
    print(df_fixed['outcome'].isna().sum())
    print(df_incorrect['outcome'].isna().sum())

    # Create the plot of the outcome distributions
    # Set all the fontsizes for the plot to be as follows
    font_size = 12
    plt.rcParams.update({'font.size': font_size - 2})
    plt.rcParams.update({'legend.fontsize': font_size - 2})
    plt.rcParams.update({'axes.labelsize': font_size - 2})
    plt.rcParams.update({'axes.titlesize': font_size})
    plt.rcParams.update({'xtick.labelsize': font_size - 4})
    plt.rcParams.update({'ytick.labelsize': font_size - 4})
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=False)

    # First subplot: Flexible ATE
    sns.kdeplot(data=df_flex,
                x='outcome',
                hue='method',
                common_norm=False,
                fill=False,
                linewidth=2.5,
                alpha=0.8,
                ax=ax[0])
    ax[0].set_title('Flexible ATE')
    ax[0].set_xlabel(r'$Y$')
    ax[0].set_ylabel(r'$P(Y)$')
    ax[0].grid(True, alpha=0.3, linestyle='--')

    # Second subplot: Fixed ATE
    sns.kdeplot(data=df_fixed,
                x='outcome',
                hue='method',
                common_norm=False,
                fill=False,
                linewidth=2.5,
                alpha=0.8,
                ax=ax[1])
    ax[1].set_title('True ATE')
    ax[1].set_xlabel(r'$Y$')
    ax[1].set_ylabel(r'$P(Y)$')
    ax[1].grid(True, alpha=0.3, linestyle='--')

    # Third subplot: Incorrect ATE
    sns.kdeplot(data=df_incorrect,
                x='outcome',
                hue='method',
                common_norm=False,
                fill=False,
                linewidth=2.5,
                alpha=0.8,
                ax=ax[2])
    ax[2].set_title('Incorrect ATE')
    ax[2].set_xlabel(r'$Y$')
    ax[2].set_ylabel(r'$P(Y)$')
    ax[2].grid(True, alpha=0.3, linestyle='--')

    # Extract unique handles and labels from the first subplot
    handles, labels = ax[0].get_legend_handles_labels()

    # Check if handles and labels are empty, and regenerate if necessary
    if not handles or not labels:
        unique_methods = df['method'].unique()
        palette = sns.color_palette('pastel')
        handles = [
            plt.Line2D([0], [0], color=palette[i], lw=10) for i in range(len(unique_methods))
        ]
        labels = unique_methods

    # Add a common legend for all subplots
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5)

    # Remove legends from individual subplots
    ax[0].legend_.remove()
    ax[1].legend_.remove()
    ax[2].legend_.remove()
    # plt.suptitle('Lalonde (Observational)', fontsize=font_size, y=0.93)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/{ds_name}_{ds_id}_{sample_size}/outcome_distribution.png',
                bbox_inches='tight',
                dpi=300)


def plot_outcome_distribution_separate(ds_name, ds_id, sample_size, setting='flexible_ate'):
    """New function: plots separate subplots for each method comparing to source for a specific setting.

    Args:
        ds_name: Dataset name (e.g., 'lalonde')
        ds_id: Dataset identifier (e.g., 'psid1')
        sample_size: Sample size (can be None)
        setting: Which setting to plot - 'flexible_ate', 'true_ate', or 'incorrect_ate'
    """
    # Create a dataframe of all the settings and methods together
    # Pick a random dataset number
    i = np.random.randint(0, 49)
    print(f'Using dataset number {i}')

    # Map setting to experiment ID
    setting_to_expt = {'flexible_ate': '0001', 'true_ate': '0002', 'incorrect_ate': '0003'}

    setting_to_label = {
        'flexible_ate': 'Flexible ATE', 'true_ate': 'True ATE', 'incorrect_ate': 'Incorrect ATE'
    }

    if setting not in setting_to_expt:
        raise ValueError(f'Setting must be one of {list(setting_to_expt.keys())}, got {setting}')

    expt_id = setting_to_expt[setting]
    setting_label = setting_to_label[setting]

    if ds_name == 'lalonde' and ds_id == 'psid1':
        source_df = load_source_df(ds_name,
                                   ds_id,
                                   sample_size,
                                   'results/GenModelCkpts/lalonde/psid1/save')
        # Load only the specific setting we want
        cred_data = load_credence_generated('lalonde', 'psid1', None, expt_id, i)
        mcred_data = load_mcredence_generated('lalonde', 'psid1', None, expt_id, i)
        rc_data = load_realcause_generated('lalonde', 'psid1', None, expt_id, i)
        ff_data = load_frugalflows_generated('lalonde', 'psid1', None, expt_id, i)
    else:
        raise ValueError(f'Dataset {ds_name} with id {ds_id} not implemented')

    # Concatenate the data
    df = pd.concat([source_df, cred_data, mcred_data, rc_data, ff_data], axis=0)
    df.reset_index(drop=True, inplace=True)

    print(df.columns)

    # Ensure that there are no nans in the outcome column
    print(f"NaNs in outcome: {df['outcome'].isna().sum()}")

    # Define the methods to plot (excluding source since we'll overlay it on each)
    methods = ['Credence', 'modCredence', 'Realcause', 'Frugalflows']
    method_labels = ['Credence', 'Modified Credence', 'RealCause', 'FrugalFlows']

    # Create the plot of the outcome distributions
    # Set all the fontsizes for the plot to be as follows
    font_size = 14
    plt.rcParams.update({'font.size': font_size - 2})
    plt.rcParams.update({'legend.fontsize': font_size - 2})
    plt.rcParams.update({'axes.labelsize': font_size - 2})
    plt.rcParams.update({'axes.titlesize': font_size})
    plt.rcParams.update({'xtick.labelsize': font_size - 4})
    plt.rcParams.update({'ytick.labelsize': font_size - 4})

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    # Plot each method in its own subplot with source as comparison
    for idx, (method, label) in enumerate(zip(methods, method_labels)):
        ax = axes[idx]

        # Get method-specific data
        method_data = df[df['method'] == method]

        # Plot source distribution (in all subplots for comparison)
        sns.kdeplot(data=source_df,
                    x='outcome',
                    color='black',
                    linewidth=2.5,
                    linestyle='--',
                    label='Source',
                    common_norm=False,
                    ax=ax)

        # Plot method distribution
        sns.kdeplot(data=method_data,
                    x='outcome',
                    linewidth=2.5,
                    label=label,
                    common_norm=False,
                    ax=ax)

        ax.set_title(f'{label}')
        ax.set_xlabel(r'$Y$')
        ax.set_ylabel(r'Density')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

    # Add overall title
    fig.suptitle(f'{setting_label}: Comparing Generated Distributions to Source',
                 fontsize=font_size + 2,
                 y=0.998)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save with setting name in filename
    filename = f'outcome_distribution_{setting}.png'
    filepath = f'{PLOTS_PATH}/{ds_name}_{ds_id}_{sample_size}/{filename}'
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f'Saved plot to {filepath}')


def plot_ate_values(ds_name, ds_id, sample_size, setting='flexible_ate'):
    """Plot the ATE values for the different methods and settings."""
    if ds_name == 'lalonde' and ds_id == 'psid1':
        # Load all datasets for all the generative methods
        all_gen_datasets = []
        for i in range(50):
            if setting == 'flexible_ate':
                cred_df = load_credence_generated('lalonde', 'psid1', None, '0001', i)
                mcred_df = load_mcredence_generated('lalonde', 'psid1', None, '0001', i)
                rc_df = load_realcause_generated('lalonde', 'psid1', None, '0001', i)
                ff_df = load_frugalflows_generated('lalonde', 'psid1', None, '0001', i)
            elif setting == 'true_ate':
                cred_df = load_credence_generated('lalonde', 'psid1', None, '0002', i)
                mcred_df = load_mcredence_generated('lalonde', 'psid1', None, '0002', i)
                rc_df = load_realcause_generated('lalonde', 'psid1', None, '0002', i)
                ff_df = load_frugalflows_generated('lalonde', 'psid1', None, '0002', i)
            elif setting == 'incorrect_ate':
                cred_df = load_credence_generated('lalonde', 'psid1', None, '0003', i)
                mcred_df = load_mcredence_generated('lalonde', 'psid1', None, '0003', i)
                rc_df = load_realcause_generated('lalonde', 'psid1', None, '0003', i)
                ff_df = load_frugalflows_generated('lalonde', 'psid1', None, '0003', i)
            else:
                raise ValueError(f'Setting {setting} not implemented')

            all_gen_datasets.append(pd.concat([cred_df, mcred_df, rc_df, ff_df], axis=0))
            all_gen_datasets[-1].reset_index(drop=True, inplace=True)

        # Join all the all_gen_datasets by adding new rows
        all_gen_datasets = pd.concat(all_gen_datasets, axis=0)
        all_gen_datasets.reset_index(drop=True, inplace=True)

        # Extract the relevant columns
        summary_df = all_gen_datasets[['method', 'dataset_num', 'ate', 'p_t']]
        # Drop duplicate rows
        summary_df = summary_df.drop_duplicates()
        # Reset the index
        summary_df.reset_index(drop=True, inplace=True)

        # Add in a row to summary_df for the source dataset
        row = ['Source', '0', 0.016, 0.0691588785046729]    # 0.41573 or 0.0691588785046729
        summary_df.loc[len(summary_df)] = row

        # Change the order of the methods
        summary_df['method'] = pd.Categorical(
            summary_df['method'], ['Source', 'Credence', 'modCredence', 'Realcause', 'FrugalFlows'])
        summary_df = summary_df.sort_values('method')

        # Create a violin plot of the 'ate' column, for each method across all dataset_nums
        plt.figure(figsize=(3, 3))
        sns.boxenplot(data=summary_df, x='method', y='ate', color='salmon')
        g = sns.stripplot(data=summary_df,
                          x='method',
                          y='ate',
                          palette='deep',
                          jitter=0.3,
                          alpha=0.8,
                          hue='method',
                          legend=False)
        # Y-ticks 45 degrees
        plt.xticks(rotation=45)
        # Add a horizontal line at 3.0
        plt.axhline(y=0.016, color='red', linestyle='--', linewidth=0.8)
        plt.xlabel('')
        plt.ylabel(r'$\tau$')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_PATH}/{ds_name}_{ds_id}_{sample_size}/ate_{setting}.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimators', type=str, default='all', choices=['class', 'all'])
    parser.add_argument('--ds_name', type=str, default=None)
    parser.add_argument('--ds_id', type=str, default=None)
    parser.add_argument('--sample_size', type=str, default=None)
    parser.add_argument('--expt_id', type=str, default=None)
    parser.add_argument('--gen_method', type=str, default=None)
    parser.add_argument('--ylims', type=float, nargs=2, required=False, default=[None, None])
    parser.add_argument('--plot_type',
                        type=str,
                        default='outcome_overlaid',
                        choices=['bias', 'outcome_overlaid', 'outcome_separate', 'ate_values'],
                        help='Type of plot to generate')
    parser.add_argument('--setting',
                        type=str,
                        default='flexible_ate',
                        choices=['flexible_ate', 'true_ate', 'incorrect_ate'],
                        help='Setting to use for outcome_separate plot',
                        required=False)
    args = parser.parse_args()

    if args.plot_type == 'bias':
        plot_bias_estimators(ds_name=args.ds_name,
                             ds_id=args.ds_id,
                             sample_size=args.sample_size,
                             expt_id=args.expt_id,
                             gen_method=args.gen_method,
                             estimators=args.estimators,
                             ylims=args.ylims,
                             regret=True)
    elif args.plot_type == 'outcome_overlaid':
        # Plot the original outcome distribution (all settings, overlaid methods)
        plot_outcome_distribution(ds_name=args.ds_name,
                                  ds_id=args.ds_id,
                                  sample_size=args.sample_size)
    elif args.plot_type == 'outcome_separate':
        # Plot the separate outcome distribution (one setting, separate subplots per method)
        plot_outcome_distribution_separate(ds_name=args.ds_name,
                                           ds_id=args.ds_id,
                                           sample_size=args.sample_size,
                                           setting=args.setting)
    elif args.plot_type == 'ate_values':
        # Plot the ATE values for the different methods and settings
        plot_ate_values(ds_name=args.ds_name,
                        ds_id=args.ds_id,
                        sample_size=args.sample_size,
                        setting=args.setting)
