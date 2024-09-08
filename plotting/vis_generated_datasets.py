"""Script to visualize characteristics of the generated datasets."""

# Import libraries
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import other modules
from data.acic2019 import load_low_dim
from data.apo import get_apo_data

# Define the constants that specify which dataset to visualize
WEIGHT = 2.5
INTERCEPT = -8.5
# DATASET_NAME = f'osapo_acic_4_weight_{WEIGHT}_intercept_{INTERCEPT}'      # pylint: disable=f-string-without-interpolation
OUTCOME_TYPE = '1_polynomial'
DATASET_NAME = f'acic_2019_{OUTCOME_TYPE}'

COUNTERFACTUAL_OUTCOMES = True
HYPERPARAM_PATH = 'dist_argsndim=10+base_distribution=uniform-dim_h64-lr0.001-batch_size16'
DATASETS_FOLDER = f'nfl_datasets/{DATASET_NAME}/{HYPERPARAM_PATH}/'
PLOTS_FOLDER = f'plots/{DATASET_NAME}/{HYPERPARAM_PATH}'


# Load the source dataset
if 'osapo_acic_4' in DATASET_NAME:
    if 'weight' in DATASET_NAME:
        source_df = get_apo_data(
            identifier='acic',
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

# Create the source dataset
source_data = source_df['w']
source_data['treatment'] = source_df['t']
source_data['outcome'] = source_df['y']

# Create the plots folder if it does not exist
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Source plot 1: Plot the distribution of the outcome variable for the two treatment groups
plt.figure()
sns.boxplot(x='treatment', y='outcome', data=source_data)
plt.xlabel('Treatment')
plt.ylabel('Outcome')
plt.title(
    'Distribution of the outcome variable for the two treatment groups in the source dataset'
)
plt.savefig(f'{PLOTS_FOLDER}/source_outcome_distribution.png',
            dpi=150)

# Source plot 2: KDE plot of the outcome variable for the two treatment groups
plt.figure()
sns.kdeplot(source_data[source_data['treatment'] == 0]['outcome'],
            label='Control')
sns.kdeplot(source_data[source_data['treatment'] == 1]['outcome'],
            label='Treatment')
plt.xlabel('Outcome')
plt.ylabel('Density')
plt.legend()
plt.title(
    'KDE plot of the outcome variable for the two treatment groups in the source dataset'
)
plt.savefig(f'{PLOTS_FOLDER}/source_outcome_kde.png', dpi=150)

# Source plot 3: Boxplots of the ITEs for the source dataset (heterogeneity)
plt.figure()
sns.boxplot(y='ites', data=source_df)
plt.ylabel('ITEs')
plt.title('Distribution of ITEs for the source dataset')
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
    # Compute the naive ATE
    naive_ate = gen_df.loc[gen_df['t'] == 1,
                           'outcome'].mean() - gen_df.loc[gen_df['t'] == 0,
                                                          'outcome'].mean()
    # Get the ITE as a column
    ites = np.array(gen_df['y1'] - gen_df['y0'])
    # Add them to the gen_df_characteristics dataframe
    gen_df_characteristics_te.loc[df_num] = [true_ate, naive_ate, ites]

# Find the proportion of treated samples in each of the generated datasets
for df_num in range(1):
    gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
                         index_col=0)
    print(gen_df['t'].value_counts(normalize=True))

# Plot 1: The distribution of true ATEs
plt.figure()
sns.boxplot(y='true_ate', data=gen_df_characteristics_te)
plt.ylabel('True ATE')
plt.title(
    'Distribution of the true ATE for the generated datasets with treatment effect'
)
plt.savefig(f'{PLOTS_FOLDER}/true_ate_distribution.png', dpi=150)

# Plot 2: Plot the distribution of the naive ATEs
plt.figure()
sns.boxplot(y='naive_ate', data=gen_df_characteristics_te)
plt.ylabel('Naive ATE')
plt.title(
    'Distribution of the naive ATE for the generated datasets with treatment effect'
)
plt.savefig(f'{PLOTS_FOLDER}/naive_ate_distribution.png', dpi=150)

# Plot 3: Plot density plots of the ITEs for each row of the gen_df_characteristics dataframe
plt.figure()
for i in range(50):
    sns.kdeplot(gen_df_characteristics_te.loc[i]['ite_distribution'])
plt.xlabel('ITE')
plt.ylabel('Density')
plt.title(
    'Density plot of ITE for the generated datasets with treatment effect (1-50)'
)
plt.savefig(f'{PLOTS_FOLDER}/ite_distribution.png', dpi=150)

# Plot 4: Overlay the density plot of the source_df with the above plot
plt.figure()
if DATASET_NAME == 'acic_2019_1_polynomial':
    # Scale the gen_df columns to ensure they have the same min and max values as the source_df
    min_val = source_data['outcome'].min()
    max_val = source_data['outcome'].max()
    gen_df['y'] = (gen_df['y'] - gen_df['y'].min()) / (gen_df['y'].max() - gen_df['y'].min()) * (max_val - min_val) + min_val
sns.kdeplot(source_data[source_data['treatment'] == 0]['outcome'],
            label='Control (Source)',
            linestyle='--')
sns.kdeplot(source_data[source_data['treatment'] == 1]['outcome'],
            label='Treatment (Source)',
            linestyle='--')
sns.kdeplot(gen_df[gen_df['t'] == 0]['y'], label='Control')
sns.kdeplot(gen_df[gen_df['t'] == 1]['y'], label='Treatment')
plt.xlabel('Outcome')
plt.ylabel('Density')
plt.legend()
plt.title(
    'KDE plot of outcome variable for the two \n treatment groups for Realcause vs. source dataset'
)
plt.savefig(f'{PLOTS_FOLDER}/observed_outcome_kde.png',
            dpi=150)

# Plot 5: Plot of the treatment variable for the source data vs. one of the generated datasets
plt.figure()
sns.kdeplot(source_data['treatment'], label='Source', linestyle='--', bw=0.1)
sns.kdeplot(gen_df['t'], label='Generated', bw=0.1)
plt.xlabel('Treatment')
plt.ylabel('Density')
plt.legend()
plt.title(
    'KDE plot of the treatment variable for the source dataset vs. one of the generated datasets'
)
plt.savefig(f'{PLOTS_FOLDER}/treatment_kde.png', dpi=150)

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
        gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
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
        plt.title('KDE plot of counterfactual outcomes for the generated datasets')
        plt.savefig(f'{PLOTS_FOLDER}/counterfactual_outcome_kde.png',
                    dpi=150)

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
        gen_df = pd.read_csv(DATASETS_FOLDER + 'dataset_' + str(df_num) + '.csv',
                            index_col=0)
        
        # Scale the following columns to ensure they have the same min and max values
        min_val = source_df['counterfactual_outcome_0'].min()
        max_val = source_df['counterfactual_outcome_0'].max()
        gen_df['y0'] = (gen_df['y0'] - gen_df['y0'].min()) / (gen_df['y0'].max() - gen_df['y0'].min()) * (max_val - min_val) + min_val
        
        min_val_1 = source_df['counterfactual_outcome_1'].min()
        max_val_1 = source_df['counterfactual_outcome_1'].max()
        gen_df['y1'] = (gen_df['y1'] - gen_df['y1'].min()) / (gen_df['y1'].max() - gen_df['y1'].min()) * (max_val_1 - min_val_1) + min_val_1
        
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
        plt.title('KDE plot of counterfactual outcomes for the generated datasets')
        plt.savefig(f'{PLOTS_FOLDER}/counterfactual_outcome_kde.png',
                    dpi=150)
            