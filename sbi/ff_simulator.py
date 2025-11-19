"""Script that uses frugal flows as the simulator for the SBI pipeline."""

# Import the necessary packages
import jax
import jax.random as jr
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

SEED = 1

def simulate_datasets(parameters,
                      sample_size,
                      dataset_identifier,
                      frugal_flow_model,
                      causal_model,
                      causal_model_args=None):
    """Function to simulate datasets using the parameters specified by the dictionary
    and using the Frugal Flow model supplied to the simulator."""
    if 'rho' in parameters:
        rho = parameters['rho']
    else:
        rho = 0.0
    if causal_model == 'location_translation':
        generated_df = frugal_flow_model.generate_samples(
            key=jr.PRNGKey(10 * SEED),
            sampling_size=sample_size,    # This is the size of the original Lalonde dataset
            copula_param=rho,    # This is the correlation parameter
            outcome_causal_model='location_translation',
            outcome_causal_args={'ate': parameters['ate']},
            with_confounding=True)
    elif causal_model == 'gaussian':
        generated_df = frugal_flow_model.generate_samples(
            key=jr.PRNGKey(10 * SEED),
            sampling_size=sample_size,    # This is the size of the original Lalonde dataset
            copula_param=rho,    # This is the correlation parameter
            outcome_causal_model='causal_cdf',
            outcome_causal_args={
                'ate': jnp.array([parameters['ate']]),
                'const': causal_model_args['const'],
                'scale': causal_model_args['scale']
            },
            with_confounding=True)
    # Depending on the dataset identifier, we may have to return specific columns
    if dataset_identifier in ['cps1','psid1']:
        generated_df.columns = [
            're78',
            'treat',
            'age',
            'education',
            're75',
            're74',
            'black',
            'hispanic',
            'married',
            'nodegree'
        ]
    return {
        'data': generated_df.values
    }
