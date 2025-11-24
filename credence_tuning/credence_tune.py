"""Script to tune the Credence models using Ray Tune.

This script tunes the outcome models for the Credence project using Ray Tune
for a specified dataset and user-defined knobs. The script is intended to
produce the best hyperparameters which can then be fed into the script `credence_data_gen.py`
"""

# Import libraries
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
# from ray.tune.search.bayesopt import BayesOptSearch # (Incompatible with tune.choice)
import yaml

# Import credence
import credence
import argparse
from credence_data_loader import load_data_credence


def tune_hyperparameters(credence_model,
                         num_epochs=1000,
                         num_trials=10,
                         outcome_model=True,
                         treatment_model=False,
                         covariates_model=False,
                         use_gpu=False,
                         experiment_identifier='',
                         exp_params=None):
    """Function to tune the hyperparameters for the model that has
    been defined for a specific dataset. The user must define
    the search space for the hyperparameters.

    Arguments:
        credence_model: Credence model object that has been defined for a specific dataset.
        outcome_model: Indicate whether or not we are training the outcome model.
        covariates_model: Indicate whether or not we are training the covariates model.
    Returns:
        best_hyperparameters: The best hyperparameters found by the hyperparameter search.
        best_metrics: The best metrics found by the hyperparameter search.
        best_result_df: The best metrics found by the hyperparameter search in a pandas dataframe.
    """
    # Define the search space of hyperparameters
    search_space = {
        'latent_dim':
            tune.choice([1, 2]),
        'hidden_dim':
            tune.choice(
                [[8, 16, 8], [4, 8, 4], [16, 32, 16], [16, 32, 64, 64, 32, 16],
                 [4, 16, 64, 64, 16,
                  4], [4, 8, 16, 32, 64, 32, 16, 8, 4]]
            ),    # [16], [8, 16, 8]]), # tune.choice([[8, 16], [16, 32], [8, 16, 8], [32, 64]]),
        'lr':
            tune.loguniform(1e-4, 5e-3),
        'kld_rigidity':
            exp_params['kld_rigidity'] if exp_params != None else tune.loguniform(0.01, 0.05),
        'bias_rigidity':
            exp_params['bias_rigidity'] if exp_params != None else tune.loguniform(500, 5000),
        'effect_rigidity':
            exp_params['effect_rigidity'] if exp_params != None else tune.loguniform(500, 5000),
        'batch_size':
            64    #tune.choice([32]),
    }
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    # Run the hyperparameter search
    if outcome_model:
        train_func = tune.with_parameters(credence_model.tune_outcome, max_epochs=num_epochs)
    if covariates_model:
        train_func = tune.with_parameters(
            credence_model.tune_covariates,
            max_epochs=num_epochs,
        )
    if treatment_model:
        train_func = tune.with_parameters(credence_model.tune_treatment, max_epochs=num_epochs)
    scaling_config = ScalingConfig(num_workers=1,
                                   use_gpu=use_gpu,
                                   resources_per_worker={
                                       'CPU': 4, 'GPU': 1 if use_gpu else 0
                                   })
    run_config = RunConfig(
        storage_path=('/scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/'
                      '/logs/credence_ray_logs'),    # No relative paths
        name=f'cred_{experiment_identifier}',
        checkpoint_config=CheckpointConfig(num_to_keep=2,
                                           checkpoint_score_attribute='val_loss',
                                           checkpoint_score_order='min'),
    )
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    # BayesOpt does not work with tune.choice
    # search_alg = BayesOptSearch(random_search_steps=4)
    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='loss',
            mode='min',
            num_samples=num_trials,
            scheduler=scheduler,
        ),
    )
    analysis = tuner.fit()

    best_result = analysis.get_best_result()
    #best_result.checkpoint.to_directory(path="")
    best_hyperparameters = best_result.config
    best_metrics = best_result.metrics
    best_result_df = best_result.metrics_dataframe
    print('Best hyperparameters found were: ', best_hyperparameters)
    print('Best metrics found were: ', best_metrics)
    print('Best result dataframe: ', best_result_df)
    return best_hyperparameters, best_metrics, best_result_df


def credence_model(dataset_name,
                   dataset_identifier=None,
                   sample_size=None,
                   experiment_identifier=None,
                   rc_model_path=None,
                   outcome_model=True,
                   covariates_model=False,
                   num_epochs=1000,
                   num_trials=10,
                   use_gpu=True,
                   use_uniform_autoencoder=False,
                   treatment_effect_fn=None,
                   effect_rigidity=None):

    with open('configs/credence_experiments.yaml', 'r', encoding='utf-8') as file:
        experiment_identifiers = yaml.safe_load(file)
    config = experiment_identifiers[f'expt_{experiment_identifier}']
    dataset_name = config['dataset_name']
    dataset_identifier = config['dataset_identifier']
    sample_size = config['sample_size']

    dataset, dataset_info = load_data_credence(dataset_name, dataset_identifier, sample_size, rc_model_path)
    # Set the value of the Credence training hyperparameters
    # We have a strong belief that there is no confounding bias
    if config['treatment_effect_val'] == 'true_ate':
        treatment_effect = dataset_info['true_ate']
    elif config['treatment_effect_val'] == 'incorrect_ate':
        treatment_effect = config['treatment_effect_fn']
    elif config['treatment_effect_val'] == 'flexible':
        treatment_effect = 0

    if outcome_model:
        # Define the Credence model
        credence_model = credence.Credence(data=dataset,
                                           post_treatment_var=dataset_info['outcome_col'],
                                           treatment_var=dataset_info['treatment_col'],
                                           categorical_var=dataset_info['categorical_vars'],
                                           numerical_var=dataset_info['continuous_vars'],
                                           treatment_effect_fn=lambda x: treatment_effect,
                                           effect_rigidity=config['effect_rigidity'],
                                           selection_bias_fn=lambda x, t: 0.0,
                                           use_uniform_encoder=use_uniform_autoencoder,
                                           use_gpu=use_gpu,
                                           bias_rigidity=config['bias_rigidity'],
                                           kld_rigidity=config['kld_rigidity'])

        # Tune the hyperparameters
        best_hyperparameters, best_metrics, best_result_df = tune_hyperparameters(
            credence_model, num_epochs = num_epochs, num_trials = num_trials,
            outcome_model = True, treatment_model = False,
            experiment_identifier=f'cred_{experiment_identifier}_outcome', exp_params = config)

        # Save the best hyperparameters and metrics to a txt file
        hyp_file = f'cred_hyperparameter_tuning/{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}'
        if treatment_effect_fn:
            hyp_file += f'_te_{dataset_info["true_ate"]}'
        #if kld_rigidity:
        #hyp_file += f'_kld_{kld_rigidity}'
        hyp_file += '_outcome'
        with open(f'{hyp_file}.txt', 'w', encoding='utf-8') as f:
            f.write(f'Best hyperparameters found were: {best_hyperparameters}\n')
            f.write(f'Best metrics found were: {best_metrics}\n')
        # Save the best hyperparameters in a pandas dataframe to a file.
        best_result_df.to_csv(f'{hyp_file}.csv')
    if covariates_model:
        # Define the Credence model
        credence_model = credence.Credence(data=dataset,
                                           post_treatment_var=dataset_info['outcome_col'],
                                           treatment_var=dataset_info['treatment_col'],
                                           categorical_var=dataset_info['categorical_vars'],
                                           numerical_var=dataset_info['continuous_vars'],
                                           generate_covariates=True,
                                           kld_rigidity=config['kld_rigidity'],
                                           use_uniform_encoder=use_uniform_autoencoder,
                                           use_gpu=use_gpu)

        # Tune the hyperparameters
        best_hyperparameters, best_metrics, best_result_df = tune_hyperparameters(
            credence_model, num_epochs = num_epochs, num_trials = num_trials,
            outcome_model=False, covariates_model=True,
            experiment_identifier=f'cred_{experiment_identifier}_covariates', exp_params = config)

        hyp_file = f'cred_hyperparameter_tuning/{dataset_name}_{dataset_identifier}_{sample_size}_expt_{experiment_identifier}_covariates'
        # Save the best hyperparameters and metrics to a txt file
        with open(f'{hyp_file}.txt', 'w', encoding='utf-8') as f:
            f.write(f'Best hyperparameters found were: {best_hyperparameters}\n')
            f.write(f'Best metrics found were: {best_metrics}\n')
        # Save the best hyperparameters in a pandas dataframe to a file.
        best_result_df.to_csv(f'{hyp_file}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default=None, required=True)
    parser.add_argument('--dataset_identifier', default=None, required=True)
    parser.add_argument('--sample_size', default=None, required=True)
    parser.add_argument('--experiment_identifier', default=None, required=True)
    parser.add_argument('--rc_model_path', default=None, required=True)
    parser.add_argument('--outcome_model',
                        type=int,
                        help='Whether to tune the outcome model.',
                        default=1,
                        required=False)
    parser.add_argument('--covariates_model',
                        type=int,
                        help='Whether to tune the covariates model.',
                        default=0,
                        required=False)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1000,
                        help='No of epochs for every trial',
                        required=False)
    parser.add_argument('--num_trials',
                        type=int,
                        default=10,
                        help='Number of hyperparameter sets to test',
                        required=False)
    parser.add_argument('--use_uniform_encoder',
                        type=int,
                        help='Whether use uniform autoencoder',
                        default=0,
                        required=False)
    parser.add_argument('--use_gpu',
                        type=int,
                        help='Variable to indicate whether or not to use gpu',
                        default=1,
                        required=False)
    args = parser.parse_args()

    if args.dataset_name == 'lalonde':
        credence_model(dataset_name=args.dataset_name,
                       dataset_identifier=args.dataset_identifier,
                       sample_size=args.sample_size,
                       experiment_identifier=args.experiment_identifier,
                       rc_model_path=args.rc_model_path,
                       outcome_model=args.outcome_model,
                       covariates_model=args.covariates_model,
                       num_epochs=args.num_epochs,
                       num_trials=args.num_trials,
                       use_gpu=args.use_gpu,
                       use_uniform_autoencoder=args.use_uniform_encoder,
                       treatment_effect_fn=None,
                       effect_rigidity=None)
    else:
        print('Invalid dataset name passed!')
        SystemExit()
