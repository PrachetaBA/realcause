"""Script to generate the sweep configuration for the hyperparameter search."""
import yaml
import argparse

parser = argparse.ArgumentParser(description='Generate the sweep configuration for the hyperparameter search.')
parser.add_argument('--dataset_name', '--dn', type=str, default=None, help='Dataset name')
parser.add_argument('--dataset_identifier', '--di', type=str, default=None, help='Dataset identifier')
parser.add_argument('--sample_size', '--ss', type=str, default=None, help='Sample size')
parser.add_argument('--causal_model', '--cm', type=str, default='location_translation', help='Causal model')
args = parser.parse_args()

DATASET_NAME = args.dataset_name
DATASET_IDENTIFIER = args.dataset_identifier
SAMPLE_SIZE = args.sample_size
CAUSAL_MODEL = args.causal_model

# Define the sweep configuration
sweep_config = {'method': 'bayes'}

metric = {'name': 'val_loss', 'goal': 'minimize'}
sweep_config['metric'] = metric

parameters_dict = {
    'learning_rate': {
        'distribution': 'uniform', 'min': 1e-5, 'max': 1e-1
    },
    'RQS_knots': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'flow_layers': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'nn_width': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'nn_depth': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'seed': {
        'distribution': 'int_uniform', 'min': 0, 'max': 100
    },
    'ate': {
        'distribution': 'uniform', 'min': -10.0, 'max': 10.0
    },
    'cm_RQS_knots': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'cm_flow_layers': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'cm_nn_width': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
    'cm_nn_depth': {
        'distribution': 'int_uniform', 'min': 1, 'max': 50
    },
}

sweep_config['parameters'] = parameters_dict

parameters_dict.update({'epochs': {'value': 1}})

sweep_config['command'] = list(
    ['${env}', '${interpreter}', '${program}', '--dataset_name', DATASET_NAME, 
     '--dataset_identifier', DATASET_IDENTIFIER, '--sample_size', SAMPLE_SIZE, '--causal_model', CAUSAL_MODEL])

with open(f'ff_hyperparameter_tuning/{DATASET_NAME}_{DATASET_IDENTIFIER}_{SAMPLE_SIZE}_{CAUSAL_MODEL}.yaml', 'w') as outfile:
    yaml.dump(sweep_config, outfile, default_flow_style=False)
