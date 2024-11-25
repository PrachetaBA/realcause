# pylint: disable=redefined-outer-name
"""Script that is used to generate datasets given the models that have 
been trained by Pracheta for the NFL paper. We will make this script 
more general - i.e. it should be able to generate datasets for the ACIC 16,
ACIC 19 as well as all the newer datasets (Kunzel, Curth, Kennedy)."""

# Import libraries
from pathlib import Path
import argparse
import numpy as np
from consts import REALCAUSE_DATASETS_FOLDER, N_AGG_SEEDS, N_SAMPLE_SEEDS
from loading import load_gen
from data.apo import get_apo_data
from data.acic2019 import load_low_dim
from data.synthetic_dgp import get_kunzel_data

def generate_datasets(gen_datasets_folder, best_model_path, data,
                      **kwargs):
    """Generates datasets given the best model path."""
    gen_datasets_folder = Path(gen_datasets_folder)
    gen_datasets_folder.mkdir(parents=True, exist_ok=True)

    # Load the best model after doing a hyperparameter search
    model, _ = load_gen(saveroot=best_model_path)
    print(f'Model is {model}')

    # Load the original dataset, so that we can use the same covariates
    if data == 'osapo_acic_4':
        outcome_type = kwargs.get('outcome_type', 'linear')
        if outcome_type == 'nonlinear':
            d = get_apo_data(identifier='acic',
                             data_format='pandas',
                             num_of_biasing_covariates=3)
            # Find the probability of treatment for this dataset
            p_t = d['t'].mean()
            print(f'Probability of treatment is {p_t}')
        else:
            weight = kwargs.get('weight', 1)
            intercept = kwargs.get('intercept', 0)
            d = get_apo_data(identifier='acic',
                         data_format='pandas',
                         weight=weight,
                         intercept=intercept,
                         num_of_biasing_covariates=1)
    elif data == 'acic2019':
        d = load_low_dim(dataset_identifier=kwargs.get('outcome_type'),
                         data_format='pandas')
    elif data == 'kunzel':
        d = get_kunzel_data(dataset_id = kwargs.get('dataset_identifier'),
                            sample_size = kwargs.get('sample_size'),
                            data_format='pandas',
                            return_ites=True,
                            return_counterfactual_outcomes=True)

    df_w, _, _ = d['w'], d['t'], d['y']
    ites = d['ites'] if 'ites' in d else None
    ate = d['ites'].mean() if 'ites' in d else None
    w_orig = df_w.to_numpy()
    print(f'Shape of the original covariates is {w_orig.shape}')
    print(f'Original ITES: {ites}')
    print(f'Original ATE: {ate}')

    dfs = []
    print(
        f'Generating {N_SAMPLE_SEEDS} datasets with {N_AGG_SEEDS} seeds for each samples'
    )

    ate_means = []
    for sample_i in range(N_SAMPLE_SEEDS):
        print('Sample:', sample_i)
        start_seed = sample_i * N_AGG_SEEDS
        end_seed = start_seed + N_AGG_SEEDS
        ates = []
        for seed in range(start_seed, end_seed):
            if 'te' in kwargs and kwargs['te'] is not None:
                _, t, (y0, y1) = model.sample(w_orig,
                                              ret_counterfactuals=True,
                                              causal_effect_scale=kwargs['te'],
                                              untransform=False,
                                              seed=seed)
            else:
                _, t, (y0, y1) = model.sample(w_orig,
                                              ret_counterfactuals=True,
                                              seed=seed)
            y = t * y1 + (1 - t) * y0
            df = df_w
            df['t'] = t
            df['y'] = y
            df['y0'] = y0
            df['y1'] = y1
            df['ite'] = y1 - y0
            ate = (y1 - y0).mean()
            ates.append(ate)

        ate_means.append(np.mean(ates))
        df.to_csv(gen_datasets_folder / f'dataset_{sample_i}.csv', index=False)
        dfs.append(df)

    print('ATEs: ', ate_means)
    print(f'ATE mean mean (min-max) (std): {np.mean(ate_means)} '
          f'({np.min(ate_means)} - {np.max(ate_means)}) '
          f'({np.std(ate_means)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate datasets given the best model path.')
    parser.add_argument('--gen_datasets_folder',
                        type=str,
                        help='Folder to save the generated datasets.')
    parser.add_argument('--best_model_path',
                        type=str,
                        help='Path to the best model.')
    parser.add_argument('--data',
                        type=str,
                        choices=['osapo_acic_4', 'acic2019', 'kunzel'],
                        help='Choice of dataset (osapo_acic_4, acic2019, kunzel)',
                        default=None)

    # Specific to the kunzel dataset
    parser.add_argument('--dataset_identifier',
                        type=str,
                        help='Additional identifier for the dataset',
                        default=None)
    parser.add_argument('--sample_size',
                        type=int,
                        help='Sample size for the kunzel dataset',
                        default=500,
                        required=False)

    # Specific to the ACIC19 dataset
    parser.add_argument('--outcome_type',
                        type=str,
                        help='Outcome type for the ACIC19 dataset',
                        default='linear',
                        required=False)
    # If outcome type is linear, specify weight and intercept.
    parser.add_argument('--weight',
                        type=float,
                        help='Weight for the covariates.',
                        default=1,
                        required=False)
    parser.add_argument('--intercept',
                        type=float,
                        help='Intercept for the covariates.',
                        default=0,
                        required=False)
    # In case the user wants to specify a fixed treatment effect
    parser.add_argument('--te',
                        type=float,
                        help='Fixed Treatment effect.',
                        default=None,
                        required=False)
    args = parser.parse_args()

    gen_datasets_folder = f'{REALCAUSE_DATASETS_FOLDER}/{args.gen_datasets_folder}'
    best_model_path = f'results/{args.best_model_path}'
    if args.data == 'osapo_acic_4':
        if args.outcome_type == 'nonlinear':
            generate_datasets(gen_datasets_folder,
                              best_model_path,
                              data=args.data,
                              outcome_type='nonlinear')
        elif args.outcome_type == 'linear':
            generate_datasets(gen_datasets_folder,
                          best_model_path,
                          data=args.data,
                          weight=args.weight,
                          intercept=args.intercept,
                          te=args.te)
    elif args.data == 'acic2019':
        generate_datasets(gen_datasets_folder,
                          best_model_path,
                          data=args.data,
                          outcome_type=args.outcome_type)
    elif args.data == 'kunzel':
        generate_datasets(gen_datasets_folder,
                          best_model_path,
                          data = args.data,
                          dataset_identifier=args.dataset_identifier,
                          sample_size=args.sample_size)

    print('Done!')
