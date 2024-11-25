"""Script to visualize characteristics of the generated datasets."""

# Import libraries
import os
import random

import argparse 
import pyjson5 as json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from KDEpy import FFTKDE, NaiveKDE
from scipy.spatial.distance import jensenshannon
import seaborn as sns

# Import other modules
from data.acic2019 import load_low_dim
from data.apo import get_apo_data
from data.synthetic_dgp import get_kunzel_data

# Find the hellinger_distance between two numpy arrays
def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)

def generate_plots(config):
    """Visualize characteristics of the generated datasets."""
    
    # Read and extract configuration parameters
    DATASET_NAME = config.get('dataset_name')
    
    if 'osapo_acic_4' in DATASET_NAME:
        if config['dataset_specific_config']['biasing'] == 'linear':
            WEIGHT = config['dataset_specific_config']['weight']
            INTERCEPT = config['dataset_specific_config']['intercept']
            DATASET_NAME = f'osapo_acic_4_weight_{WEIGHT}_intercept_{INTERCEPT}' 
        elif config['dataset_specific_config']['biasing'] == 'nonlinear':
            DATASET_NAME = f'osapo_acic_4_nonlinear_3cov'
    elif 'acic_2019' in DATASET_NAME:
        OUTCOME_TYPE = config['dataset_specific_config']['outcome_type']
        DATASET_NAME = f'acic_2019_{OUTCOME_TYPE}'
    elif 'kunzel' in DATASET_NAME:
        DATASET_IDENTIFIER = config['dataset_specific_config']['dataset_identifier']
        SAMPLE_SIZE = config['dataset_specific_config']['sample_size']
        DATASET_NAME = f'kunzel_{DATASET_IDENTIFIER}_ss_{SAMPLE_SIZE}'
    
    DATASETS_FOLDER = f"{config.get('dataset_folder')}/{DATASET_NAME}/"
    PLOTS_FOLDER = f"{config.get('plots_folder')}/{DATASET_NAME}"
    HYPERPARAM_PATH = config.get('hyperparam_path')
    COUNTERFACTUAL_OUTCOMES = config.get('plot_counterfactuals')

    # Load the source dataset
    if 'osapo_acic_4' in DATASET_NAME:
        if 'weight' in DATASET_NAME:
            source_df = get_apo_data(identifier='acic',
                                    data_format='pandas',
                                    num_of_biasing_covariates=1,
                                    ret_counterfactual_outcomes=False,
                                    weight=WEIGHT,
                                    intercept=INTERCEPT)
        elif 'nonlinear' in DATASET_NAME:
            source_df = get_apo_data(identifier='acic',
                                    data_format='pandas',
                                    num_of_biasing_covariates=3)
    elif 'acic_2019' in DATASET_NAME:
        source_df = load_low_dim(dataset_identifier=OUTCOME_TYPE,
                                data_format='pandas')
    elif 'kunzel' in DATASET_NAME:
        source_df = get_kunzel_data(dataset_id=DATASET_IDENTIFIER,
                                    sample_size=SAMPLE_SIZE,
                                    data_format='pandas',
                                    return_ites=True)

    # Create the source dataset
    source_data = source_df['w']
    source_data['treatment'] = source_df['t']
    source_data['outcome'] = source_df['y']
    source_ate = np.mean(source_df['ites'])
    naive_ate = source_data[source_data['treatment'] == 1]['outcome'].mean() - source_data[source_data['treatment'] == 0]['outcome'].mean()
    print(f'Source ATE: {source_ate}')
    print(f'Source naive ATE: {naive_ate}')

    # Create the plots folder if it does not exist
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    # Source plot 1: Plot the distribution of the outcome variable for the two treatment groups
    plt.figure()
    sns.boxplot(x='treatment', y='outcome', data=source_data)
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.title(
        r'Source data (Observed): $Y|T=1, Y|T=0$'
    )
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/source_outcome_boxplot.png', dpi=150)

    # Source plot 2: KDE plot of the outcome variable for the two treatment groups
    plt.figure()
    sns.kdeplot(source_data[source_data['treatment'] == 0]['outcome'],
                label='T = 0')
    sns.kdeplot(source_data[source_data['treatment'] == 1]['outcome'],
                label='T = 1')
    plt.xlabel('Outcome')
    plt.ylabel('Density')
    plt.legend()
    plt.title(
        r'Source data (Observed): $P(Y|T=1), P(Y|T=0)$'
    )
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/source_outcome_kde.png', dpi=150)

    # Source plot 3: Boxplots of the ITEs for the source dataset (heterogeneity)
    plt.figure()
    sns.boxplot(y='ites', data=source_df)
    plt.ylabel('ITEs')
    plt.title(r'Source data ITE distribution: $P(Y(1) - Y(0))$')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/source_ite_distribution.png', dpi=150)

    gen_df_characteristics_te = pd.DataFrame(
        columns=['true_ate', 'naive_ate', 'ite_distribution'])
    for df_num in range(50):
        # Read in the generated dataset
        gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
                            index_col=0)
        # Create the outcome column
        gen_df['outcome'] = gen_df['t'] * gen_df['y1'] + (
            1 - gen_df['t']) * gen_df['y0']
        # Compute the true ATE
        true_ate = (gen_df['y1'] - gen_df['y0']).mean()
        # Compute the naive ATE (with confounder bias)
        naive_ate = gen_df.loc[gen_df['t'] == 1,
                            'outcome'].mean() - gen_df.loc[gen_df['t'] == 0,
                                                            'outcome'].mean()
        # Get the ITE as a column
        ites = np.array(gen_df['y1'] - gen_df['y0'])
        # Add them to the gen_df_characteristics dataframe
        gen_df_characteristics_te.loc[df_num] = [true_ate, naive_ate, ites]

    # Find the proportion of treated samples in each of the generated datasets
    # Calculate propensity score for the source dataset 
    source_propensity_score = source_data['treatment'].value_counts(normalize=True)
    propensity_scores = []
    for df_num in range(50):
        gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
                            index_col=0)
        propensity_scores.append(gen_df['t'].value_counts(normalize=True)[0])
    # Create a boxplot of the propensity scores for the generated datasets 
    plt.figure()
    plt.boxplot(np.array(propensity_scores))
    plt.scatter(1, source_propensity_score[0], color='red', label=r'Source $\pi(X)$')
    plt.xlabel('P(T=1)')
    plt.title(r'Distribution of $\pi(X)$ for the generated datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/gen_propensity_scores.png', dpi=150)

    # Distribution of ATEs for the generated datasets. 
    plt.figure()
    sns.boxplot(y='true_ate', data=gen_df_characteristics_te)
    plt.scatter(0, source_ate, color='red', label='Source ATE')
    plt.ylabel('True ATE')
    plt.title(
        'Distribution of ATEs for the generated datasets'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/gen_ate_distribution.png', dpi=150)

    # Plot 2: Plot the distribution of the naive ATEs
    plt.figure()
    sns.boxplot(y='naive_ate', data=gen_df_characteristics_te)
    plt.scatter(0, naive_ate, color='red', label='Source Naive ATE')
    plt.ylabel('Naive ATE (Bias)')
    plt.title(
        'Distribution of bias for the generated datasets'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/gen_naive_ate_distribution.png', dpi=150)

    # Plot 3: Plot density plots of the ITEs for each row of the gen_df_characteristics dataframe
    plt.figure()
    for i in range(50):
        sns.kdeplot(gen_df_characteristics_te.loc[i]['ite_distribution'])
    plt.xlabel('ITE')
    plt.ylabel('Density')
    plt.title(
        'Density plot of ITEs for the generated datasets'
    )
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/gen_ite_distribution.png', dpi=150)

    # Plot 4: Overlay the density plot of the source_df with the above plot
    plt.figure()
    if DATASET_NAME == 'acic_2019_1_polynomial':
        # Scale the gen_df columns to ensure they have the same min and max values as the source_df
        min_val = source_data['outcome'].min()
        max_val = source_data['outcome'].max()
        gen_df['y'] = (gen_df['y'] - gen_df['y'].min()) / (
            gen_df['y'].max() - gen_df['y'].min()) * (max_val - min_val) + min_val
    sns.kdeplot(source_data[source_data['treatment'] == 0]['outcome'],
                label='Y|T = 0 (Source)',
                linestyle='--')
    sns.kdeplot(source_data[source_data['treatment'] == 1]['outcome'],
                label='Y|T = 1 (Source)',
                linestyle='--')
    sns.kdeplot(gen_df[gen_df['t'] == 0]['y'], label='Y|T = 0 (RC)')
    sns.kdeplot(gen_df[gen_df['t'] == 1]['y'], label='Y|T = 1 (RC)')
    plt.xlabel(r'$Y$')
    plt.ylabel(r'$P(Y|T)$')
    plt.legend()
    plt.title(
        'Distribution of outcome for Realcause vs. Source (single gen dataset)'
    )
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/gen_outcome_distribution.png', dpi=150)

    # Plot 5: Plot of the treatment variable for the source data vs. one of the generated datasets
    plt.figure()
    sns.kdeplot(source_data['treatment'], label='Source', linestyle='--', bw=0.1)
    sns.kdeplot(gen_df['t'], label='Generated', bw=0.1)
    plt.xlabel('Treatment')
    plt.ylabel('Density')
    plt.legend()
    plt.title(
        'Distribution of treatment for Realcause vs. Source (single gen dataset)'
    )
    plt.tight_layout()
    plt.savefig(f'{PLOTS_FOLDER}/gen_treatment_distribution.png', dpi=150)

    # # Find the Hellinger distance between the observed outcome distributions for both the control and treatment groups
    # # for all 50 of the generated datasets, then find the mean JSD
    # hd_observed_outcome_control = []
    # hd_observed_outcome_treatment = []
    # hd_treatment = []
    # for df_num in range(50):
    #     gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
    #                          index_col=0)
    #     # Compute the KDE distribution for the outcome variable for the control and treatment groups
    #     x, y = NaiveKDE(bw='ISJ').fit(source_data[source_data['treatment'] == 0]['outcome']).evaluate()
    #     x_gen, y_gen = NaiveKDE(bw='ISJ').fit(gen_df[gen_df['t'] == 0]['y']).evaluate()
    #     hd_observed_outcome_control.append(hellinger_distance(y, y_gen))

    #     x, y = NaiveKDE(bw=0.5, kernel='epa').fit(source_data[source_data['treatment'] == 1]['outcome']).evaluate()
    #     x_gen, y_gen = NaiveKDE(bw=0.5, kernel='epa').fit(gen_df[gen_df['t'] == 1]['y']).evaluate()
    #     hd_observed_outcome_treatment.append(hellinger_distance(y, y_gen))

    #     # Compute the Hellinger distance between the treatment variable for the source and generated datasets
    #     hd_treatment.append(hellinger_distance(source_data['treatment'],
    #                                              gen_df['t']))
    # print('HD observed outcomes: ', hd_observed_outcome)
    # print(f'Mean Hellinger between observed outcomes: {np.mean(hd_observed_outcome)}')
    # print(f'Mean Hellinger between treatment variable: {np.mean(hd_treatment)}')

    if COUNTERFACTUAL_OUTCOMES:
        if 'osapo_acic_4' in DATASET_NAME:
            if 'weight' in DATASET_NAME:
                # Changes the source dataframe
                source_df = get_apo_data(identifier='acic',
                                        data_format='pandas',
                                        num_of_biasing_covariates=1,
                                        ret_counterfactual_outcomes=True,
                                        weight=WEIGHT,
                                        intercept=INTERCEPT)
            else:
                source_df = get_apo_data(identifier='acic',
                                        data_format='pandas',
                                        num_of_biasing_covariates=3,
                                        ret_counterfactual_outcomes=True)
            # Plot 6: Plot the counterfactual outcomes for the source data for any generated dataset
            df_num = random.randint(0, 49)
            print(f'Randomly picked dataset number: {df_num}')
            gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) +
                                '.csv',
                                index_col=0)
            plt.figure()
            sns.kdeplot(
                source_df[source_df['treatment'] == 0]['counterfactual_outcome_1'],
                label='Control (Source Counterfactual)',
                linestyle='--')
            sns.kdeplot(
                source_df[source_df['treatment'] == 1]['counterfactual_outcome_0'],
                label='Treatment (Source Counterfactual)',
                linestyle='--')
            sns.kdeplot(gen_df[gen_df['t'] == 0]['y1'],
                        label='Control (Counterfactual)')
            sns.kdeplot(gen_df[gen_df['t'] == 1]['y0'],
                        label='Treatment (Counterfactual)')
            plt.xlabel('Counterfactual Outcome')
            plt.ylabel('Density')
            plt.legend()
            plt.title(
                'KDE plot of counterfactual outcomes for the generated datasets')
            plt.savefig(f'{PLOTS_FOLDER}/counterfactual_outcome_kde.png', dpi=150)

        elif 'acic_2019' in DATASET_NAME:
            # Change this function to include a way to return counterfactual outcomes
            df = load_low_dim(dataset_identifier=OUTCOME_TYPE,
                            data_format='pandas',
                            return_counterfactual_outcomes=True)
            # Convert dict to dataframe
            source_df = df['w']
            # Add treatment and outcome columns
            source_df['treatment'] = df['t']
            source_df['outcome'] = df['y']
            source_df['counterfactual_outcome_0'] = df['counterfactual_outcomes_0']
            source_df['counterfactual_outcome_1'] = df['counterfactual_outcomes_1']

            df_num = random.randint(0, 49)
            print(f'Randomly picked dataset number: {df_num}')
            gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) +
                                '.csv',
                                index_col=0)

            # Scale the following columns to ensure they have the same min and max values
            min_val = source_df['counterfactual_outcome_0'].min()
            max_val = source_df['counterfactual_outcome_0'].max()
            gen_df['y0'] = (gen_df['y0'] - gen_df['y0'].min()) / (gen_df['y0'].max(
            ) - gen_df['y0'].min()) * (max_val - min_val) + min_val

            min_val_1 = source_df['counterfactual_outcome_1'].min()
            max_val_1 = source_df['counterfactual_outcome_1'].max()
            gen_df['y1'] = (gen_df['y1'] - gen_df['y1'].min()) / (gen_df['y1'].max(
            ) - gen_df['y1'].min()) * (max_val_1 - min_val_1) + min_val_1

            # Plot 6: Plot the counterfactual outcomes for the source data for any generated dataset
            plt.figure()
            sns.kdeplot(
                source_df[source_df['treatment'] == 0]['counterfactual_outcome_1'],
                label='Control (Source Counterfactual)',
                linestyle='--')
            sns.kdeplot(
                source_df[source_df['treatment'] == 1]['counterfactual_outcome_0'],
                label='Treatment (Source Counterfactual)',
                linestyle='--')
            sns.kdeplot(gen_df[gen_df['t'] == 0]['y1'],
                        label='Control (Counterfactual)')
            sns.kdeplot(gen_df[gen_df['t'] == 1]['y0'],
                        label='Treatment (Counterfactual)')
            plt.xlabel('Counterfactual Outcome')
            plt.ylabel('Density')
            plt.legend()
            plt.title(
                'KDE plot of counterfactual outcomes for the generated datasets')
            plt.savefig(f'{PLOTS_FOLDER}/counterfactual_outcome_kde.png', dpi=150)
        elif 'kunzel' in DATASET_NAME:
            df = get_kunzel_data(dataset_id=DATASET_IDENTIFIER,
                                sample_size=SAMPLE_SIZE,
                                data_format='pandas',
                                return_ites=True,
                                return_counterfactual_outcomes=True)
            source_df = df['w']
            source_df['treatment'] = df['t']
            source_df['outcome'] = df['y']
            source_df['counterfactual_outcome_0'] = df['counterfactual_outcomes_0']
            source_df['counterfactual_outcome_1'] = df['counterfactual_outcomes_1']

            df_num = random.randint(0, 49)
            print(f'Randomly picked dataset number: {df_num}')
            gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) +
                                '.csv',
                                index_col=0)

            # Scale the following columns to ensure they have the same min and max values
            min_val = source_df['counterfactual_outcome_0'].min()
            max_val = source_df['counterfactual_outcome_0'].max()
            gen_df['y0'] = (gen_df['y0'] - gen_df['y0'].min()) / (gen_df['y0'].max(
            ) - gen_df['y0'].min()) * (max_val - min_val) + min_val

            min_val_1 = source_df['counterfactual_outcome_1'].min()
            max_val_1 = source_df['counterfactual_outcome_1'].max()
            gen_df['y1'] = (gen_df['y1'] - gen_df['y1'].min()) / (gen_df['y1'].max(
            ) - gen_df['y1'].min()) * (max_val_1 - min_val_1) + min_val_1

            # Plot 6: Plot the counterfactual outcomes for the source data for any generated dataset
            plt.figure()
            sns.kdeplot(
                source_df[source_df['treatment'] == 0]['counterfactual_outcome_1'],
                label='Y|T=0 (Source CF)',
                linestyle='--')
            sns.kdeplot(
                source_df[source_df['treatment'] == 1]['counterfactual_outcome_0'],
                label='Y|T=1 (Source CF)',
                linestyle='--')
            sns.kdeplot(gen_df[gen_df['t'] == 0]['y1'],
                        label='Y|T=0 (RC CF)')
            sns.kdeplot(gen_df[gen_df['t'] == 1]['y0'],
                        label='Y|T=1 (RC CF)')
            plt.xlabel('Counterfactual Outcome')
            plt.ylabel('Density')
            plt.legend()
            plt.title(
                'Distribution of counterfactual outcome for RC vs. Source (single gen dataset)')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_FOLDER}/gen_cf_outcome_distribution.png', dpi=150)

            # Create a plot of the density of counterfactual_outcome_1 and counterfactual_outcome_0 for ~10 different generated datasets as well as the source dataset. Do this side by side
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            for i in range(0, 50, 10): 
                gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(i) + '.csv',
                                    index_col=0)
                sns.kdeplot(
                    gen_df[gen_df['t'] == 0]['y1'],
                    label=f'Y|T=0 (RC) {i}',
                    ax=ax[0])
                sns.kdeplot(
                    gen_df[gen_df['t'] == 1]['y0'],
                    label=f'Y|T=1 (RC) {i}',
                    ax=ax[1])
            sns.kdeplot(
                source_df[source_df['treatment'] == 0]['counterfactual_outcome_1'],
                label='Y|T=0 (Source)',
                linestyle='--',
                ax=ax[0])
            sns.kdeplot(
                source_df[source_df['treatment'] == 1]['counterfactual_outcome_0'],
                label='Y|T=1 (Source)',
                linestyle='--',
                ax=ax[1])
            ax[0].set_xlabel('Counterfactual Outcome')
            ax[0].set_ylabel('Density')
            ax[0].legend()
            ax[0].set_title('Counterfactual Outcome for Control')
            ax[1].set_xlabel('Counterfactual Outcome')
            ax[1].legend()
            ax[1].set_title('Counterfactual Outcome for Treatment')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_FOLDER}/gen_cf_outcome_distribution_multiple.png', dpi=150)

    # For all the generated datasets, find the JSD between the counterfactual outcomes
    # for the control and treatment groups
    # hd_counterfactual_outcome = []
    # for df_num in range(1):
    #     gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
    #                          index_col=0)
    #     # Combine all the counterfactual outcomes for the control and treatment groups
    #     source_cf_outcomes = np.concatenate(
    #         (source_df[source_df['treatment'] == 0]['counterfactual_outcome_1'],
    #          source_df[source_df['treatment'] == 1]['counterfactual_outcome_0']))
    #     gen_cf_outcomes = np.concatenate(
    #         (gen_df[gen_df['t'] == 0]['y1'], gen_df[gen_df['t'] == 1]['y0']))
    #     print(f'Source cf outcomes: {source_cf_outcomes}')
    #     print(f'Gen cf outcomes: {gen_cf_outcomes}')
    #     hd_counterfactual_outcome.append(
    #         hellinger_distance(source_cf_outcomes, gen_cf_outcomes))
    # print(
    #     f'Mean Hellinger distance between counterfactual outcomes: {np.mean(hd_counterfactual_outcome)}'
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize generated datasets')
    parser.add_argument('--config_file',
                        '-c',
                        type=str,
                        default='gen_datasets.jsonc',
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        
    generate_plots(config)