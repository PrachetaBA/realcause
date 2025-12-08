"""Script to tune hyperparameters using Comet ML experiments."""

import argparse
import ast
import os
from collections import OrderedDict
import json

import numpy as np
import comet_ml
import torch
import gpytorch
import yaml

from data_loaders import apo
from models import TarNet, preprocess, TrainingParams, MLPParams, LinearModel, GPModel, TarGPModel, GPParams
from models import distributions
import helpers

# from utils import get_duplicates


def get_data(args):
    """Function to extract the specific data in the format required for Realcause."""
    data_name = args.data.lower()
    data_id = args.data_identifier
    ate = None
    ite = None
    if data_name in ['n_acic_4', 'jdk', 'postgres']:
        d, _ = apo.get_apo_data(identifier=data_name, confound_func=data_id,
                         data_format='numpy', return_ites=True,
                         ret_counterfactual_outcomes=False,
                         sample_size=args.sample_size)
        w, t, y = d['w'], d['t'], d['y']
        ite = d['ite'] if 'ite' in d else None
        ate = d['ite'].mean() if 'ite' in d else None
    else:
        raise ValueError(f'Dataset {data_name} not implemented')
    return w, t, y, ite, ate


def get_distribution(dist_name, dist_args=None, atoms=None):
    """
    args.dist_args should be a list of keyword:value pairs.

      examples:
      1) ['ndim:5']
      2) ['ndim:10', 'base_distribution:uniform']
    """
    if dist_args:
        dist_args = ast.literal_eval(dist_args)
    kwargs = dict()
    if len(dist_args) > 0:
        for a in dist_args:
            k, v = a.split('=')
            if v.isdigit():
                v = int(v)
            kwargs.update({k: v})

    if dist_name in distributions.BaseDistribution.dist_names:
        dist = distributions.BaseDistribution.dists[dist_name](**kwargs)
    else:
        raise NotImplementedError(
            f'Got dist argument `{dist_name}`, not one of {distributions.BaseDistribution.dist_names}'
        )
    if atoms:
        dist = distributions.MixedDistribution(atoms, dist)
    return dist


def evaluate(num_univariate_tests, model):
    all_runs = list()
    t_pvals = list()
    y_pvals = list()

    for _ in range(num_univariate_tests):
        uni_metrics = model.get_univariate_quant_metrics(dataset='test')
        all_runs.append(uni_metrics)
        t_pvals.append(uni_metrics['t_ks_pval'])
        y_pvals.append(uni_metrics['y_ks_pval'])

    summary = OrderedDict()

    summary.update(nll=model.best_val_loss)
    summary.update(avg_t_pval=sum(t_pvals) / num_univariate_tests)
    summary.update(avg_y_pval=sum(y_pvals) / num_univariate_tests)
    summary.update(min_t_pval=min(t_pvals))
    summary.update(min_y_pval=min(y_pvals))
    summary.update(q30_t_pval=np.percentile(t_pvals, 30))
    summary.update(q30_y_pval=np.percentile(y_pvals, 30))
    summary.update(q50_t_pval=np.percentile(t_pvals, 50))
    summary.update(q50_y_pval=np.percentile(y_pvals, 50))

    summary.update(ate_exact=model.ate().item())
    summary.update(ate_noisy=model.noisy_ate().item())

    return summary, all_runs


def main(args, save_args=True, log_=True):
    # create logger
    helpers.create(*args.saveroot.split('/'))
    logger = helpers.Logging(args.saveroot, 'log.txt', log_)
    logger.info(args)

    comet_exp_name = f"{args.saveroot.split('/')[-1]}"

    # save args
    if save_args:
        with open(os.path.join(args.saveroot, 'args.txt'), 'w') as file:
            file.write(json.dumps(args.__dict__, indent=4))

    # dataset
    logger.info(f'getting data: {args.data}')
    w, t, y, ite, ate = get_data(args)

    # Debugging
    if args.verbose:
        logger.debug(f'w: {w.shape}, t: {t.shape}, y: {y.shape}')
        logger.debug(f'w: {w[:5]}, t: {t[:5]}, y: {y[:5]}')
        logger.debug(f'ITEs: {ite.shape}, ATE: {ate}')

    # Regardless, print the average treatment effect.
    logger.info(f'ate: {ate}')

    # comet login - initialize the project
    comet_ml.login(project_name=f'realcause-{comet_exp_name}', api_key='FZHDy6k24i2GOKtzc85PjAPNY')

    # Read the hyperparameter file and create the optimizer, if we have only one set of
    # values in the hyperparameter file, then we will treat it as a fixed parameter
    # else, it will be tuned in the optimizer
    with open(f'hyperparameter_tuning/{args.hyperparameter_file}.yaml', 'r',
              encoding='utf-8') as file:
        tuning_config = yaml.safe_load(file)

    hps = tuning_config['hps']
    model_parameters = tuning_config['model_parameters']
    spec = tuning_config['spec']
    logger.info(f'Hyperparameter file: {args.hyperparameter_file}')
    logger.info(f'Hyperparameters: {hps}')
    logger.info(f'Model parameters: {model_parameters}')
    logger.info(f'Spec: {spec}')

    optimizer_config = {
        'algorithm': 'bayes',
        'spec': spec,
        'parameters': model_parameters,
        'name': f'BayesOpt_{comet_exp_name}',
        'trials': 1,
    }

    # For the record, let us print the full set of hyperparameters and fixed parameters that
    # are used for training Realcause generator.
    logger.info('#' * 80)
    logger.info(
        'Recording the set of fixed and variable hyperparameters used for training the model.')
    logger.info(f'Fixed hyperparameters: {hps}')
    logger.info(f'Variable hyperparameters: {model_parameters}')
    logger.info(f'Optimizer specification: {spec}')
    logger.info(f'Optimizer configuration: {optimizer_config}')
    logger.info('#' * 80)

    # Initialize the optimizer
    opt = comet_ml.Optimizer(config=optimizer_config)

    # Run each experiment
    for experiment in opt.get_experiments():
        experiment.add_tag(args.data)

        # distribution of outcome (y)
        distribution = get_distribution(dist_name=experiment.get_parameter('dist'),
                                        dist_args=experiment.get_parameter('dist_args'),
                                        atoms=hps['atoms'])
        logger.info(distribution)

        # training params
        training_params = TrainingParams(
            lr=experiment.get_parameter('lr'),
            batch_size=experiment.get_parameter('batch_size'),
            num_epochs=hps['num_epochs'],
        )
        logger.info(training_params.__dict__)

        # initializing model
        w_transform = preprocess.Preprocess.preps[hps['w_transform']]
        y_transform = preprocess.Preprocess.preps[hps['y_transform']]
        outcome_min = 0 if hps['y_transform'] == 'Normalize' else None
        outcome_max = 1 if hps['y_transform'] == 'Normalize' else None

        # model type
        additional_args = dict()
        if hps['model_type'] == 'tarnet':
            Model = TarNet

            logger.info('model type: tarnet')
            mlp_params = MLPParams(
            # n_hidden_layers=hps['n_hidden_layers'],   # Fixed
                n_hidden_layers=experiment.get_parameter('n_hidden_layers'),    # Variable
                dim_h=experiment.get_parameter('dim_h'),
                activation=getattr(torch.nn, hps['activation'])(),
            )
            logger.info(mlp_params.__dict__)
            network_params = dict(
                mlp_params_w=mlp_params,
                mlp_params_t_w=mlp_params,
                mlp_params_y0_w=mlp_params,
                mlp_params_y1_w=mlp_params,
            )
        elif hps['model_type'] == 'linear':
            Model = LinearModel

            logger.info('model type: linear model')
            network_params = dict()
        elif 'gp' in hps['model_type']:
            if hps['model_type'] == 'gp':
                Model = GPModel
            elif hps['model_type'] == 'targp':
                Model = TarGPModel
            else:
                raise ValueError(f'model type {hps["model_type"]} not implemented')
            logger.info('model type: linear model')

            kernel_t = gpytorch.kernels.__dict__[hps['kernel_t']]()
            kernel_y = gpytorch.kernels.__dict__[hps['kernel_y']]()
            var_dist = gpytorch.variational.__dict__[hps['var_dist']]
            network_params = dict(
                gp_t_w=GPParams(kernel=kernel_t, var_dist=var_dist),
                gp_y_tw=GPParams(kernel=kernel_y, var_dist=None),
            )
            logger.info(f'gp_t_w: {repr(network_params["gp_t_w"])}'
                        f'gp_y_tw: {repr(network_params["gp_y_tw"])}')
            additional_args['num_tasks'] = hps['num_tasks']
        else:
            raise ValueError(f'model type {hps["model_type"]} not implemented')

        if args.verbose:
            logger.debug(f'Initialized the network to be {network_params}')

        # Create unique savepath for each experiment to avoid loading checkpoints with mismatched hyperparameters
        experiment_savepath = os.path.join(args.saveroot, f'model_{experiment.id}.pt')

        model = Model(
            w,
            t,
            y,
            training_params=training_params,
            network_params=network_params,
            binary_treatment=True,
            outcome_distribution=distribution,
            outcome_min=outcome_min,
            outcome_max=outcome_max,
            train_prop=hps['train_prop'],
            val_prop=hps['val_prop'],
            test_prop=hps['test_prop'],
            seed=hps['seed'],
            early_stop=hps['early_stop'],
            patience=hps['patience'],
            ignore_w=hps['ignore_w'],
            grad_norm=hps['grad_norm'],
            w_transform=w_transform,
            y_transform=y_transform,    # TODO set more args
            savepath=experiment_savepath,
            test_size=hps['test_size'],
            additional_args=additional_args)

        # TODO: Add GPU support later
        if args.train:
            model.train(print_=logger.info, comet_exp=experiment)

        # End the current experiment
        experiment.end()

    # After the ideal hyperparameters have been found, run the evaluation
    if args.eval:
        summary, all_runs = evaluate(hps['num_univariate_tests'], model)
        logger.info(summary)
        with open(os.path.join(args.saveroot, 'summary.txt'), 'w') as file:
            file.write(json.dumps(summary, indent=4))
        with open(os.path.join(args.saveroot, 'all_runs.txt'), 'w') as file:
            file.write(json.dumps(all_runs))

        model.plot_ty_dists()


def get_args():
    parser = argparse.ArgumentParser(description='causal-gen')

    # dataset
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--data_identifier', type=str, default=None, required=False)
    parser.add_argument('--saveroot', type=str, default='tuned_models')
    parser.add_argument('--hyperparameter_file', type=str,
                        default=None)    # Name of the hyperparameter file to use for tuning
    parser.add_argument('--train', type=eval, default=True, choices=[True, False])
    parser.add_argument('--eval', type=eval, default=False, choices=[True, False])
    parser.add_argument('--overwrite_reload',
                        type=str,
                        default='',
                        help='secondary folder name of an experiment')    # TODO: for model loading
    # logging level
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--sample_size', type=str, default=None,
                        required=False)    # To pick sample size of dataset
    return parser


if __name__ == '__main__':
    print(f'Starting to run {__file__} now.')
    main(get_args().parse_args())
