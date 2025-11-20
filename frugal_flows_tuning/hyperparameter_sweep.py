"""Script to search for optimal hyperparameters for the Frugal Flows model."""

# Add workspace root to Python path to enable importing frugal_flows
import sys
import os

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import libraries
import argparse
import jax
import jax.numpy as jnp
import fit_frugal_models as ff_fit
import wandb
from ff_data_loader import load_data_ff

# Some fixed settings
jnp.set_printoptions(precision=2)
jax.config.update('jax_enable_x64', True)

# Log which platform JAX is using
print(f"JAX platform: {jax.devices()}")

def train(dataset_name, config=None, dataset_identifier=None, sample_size=None, causal_model=None, rc_model_path=None):
    """Function to train the Frugal flow model on the data defined by the arguments."""
    # Load the data, and extract the JNP arrays
    data = load_data_ff(dataset_name, dataset_identifier, sample_size, rc_model_path)
    X = data['X']
    Y = data['Y']
    Z_disc = data['Z_disc']
    Z_cont = data['Z_cont']

    # Run the sweep
    with wandb.init(config=config):
        config = wandb.config

        hyperparams_dict = {
            'learning_rate': config.learning_rate,
            'RQS_knots': config.RQS_knots,
            'flow_layers': config.flow_layers,
            'nn_width': config.nn_width,
            'nn_depth': config.nn_depth,
            'max_patience': 100,
            'max_epochs': 10000,    # 20000 original
        }
        if causal_model == 'location_translation':
            causal_margin_hyperparams_dict = {
                'RQS_knots': config.cm_RQS_knots,
                'flow_layers': config.cm_flow_layers,
                'nn_width': config.cm_nn_width,
                'nn_depth': config.cm_nn_depth,
            }
            output, min_loss = ff_fit.frugal_fitting(
                X=X,
                Y=Y,
                Z_disc=Z_disc,
                Z_cont=Z_cont,
                seed=config.seed,
                frugal_flow_hyperparams=hyperparams_dict,
                causal_model='location_translation',
                causal_model_args={'ate': 0.0, **causal_margin_hyperparams_dict})
        elif causal_model == 'gaussian':
            causal_margin_hyperparams_dict = {'ate': jnp.array([0.]), 'const': 0.0, 'scale': 1}
            output, min_loss = ff_fit.frugal_fitting(
                X=X,
                Y=Y,
                Z_disc=Z_disc,
                Z_cont=Z_cont,
                seed=config.seed,
                frugal_flow_hyperparams=hyperparams_dict,
                causal_model='gaussian',
                causal_model_args={'ate': causal_margin_hyperparams_dict['ate'],
                                   'const': causal_margin_hyperparams_dict['const'],
                                   'scale': causal_margin_hyperparams_dict['scale']})
        else: 
            raise ValueError('Causal model not recognized, unable to extract causal margin')
        print('Learned causal margin (ATE) is: ', output['causal_margin'])
        print('Loss is: ', min_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frugal Flows hyperparameter sweep')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    parser.add_argument('--dataset_identifier', type=str, default=None, help='Dataset identifier')
    parser.add_argument('--sample_size', type=str, default=None, help='Sample size to use')
    parser.add_argument('--causal_model', type=str, default=None, help='Causal model to use')
    args = parser.parse_args()
    train(dataset_name=args.dataset_name,
          config=None,
          dataset_identifier=args.dataset_identifier,
          sample_size=args.sample_size,
          causal_model=args.causal_model)
