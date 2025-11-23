"""Script to perform SMC-ABC for the SBICE pipeline
using both the Realcause and FrugalFlows simulator
such that we can perform model selection and posterior
estimation simultaneously."""

# Import libraries
import argparse
import logging
import os 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
import yaml

import jax
import jax.random as jr
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from loading import load_gen
from data_loaders import lalonde as rc_lalonde # Data loaders for Realcause simulator
from data_loaders import twins as rc_twins # Data loaders for Twins simulator
from sbi import rc_simulator  # Realcause simulator
from sbi import ff_simulator # FrugalFlows simulator
from frugal_flows.benchmarking import FrugalFlowModel

# Defing logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear any cached GPU state to avoid CUDA_ERROR_ILLEGAL_ADDRESS
# This helps when GPU is in a corrupted state from previous jobs in Slurm
try:
    # Try to get devices to ensure proper initialization
    # This will fail early if GPU is corrupted, allowing better error messages
    devices = jax.devices()
    logger.info(f"JAX initialized with devices: {devices}")
    
    # Try a simple operation to test GPU state
    test_array = jnp.array([1.0, 2.0, 3.0])
    _ = jnp.sum(test_array)
    logger.info("JAX GPU test operation successful")
except Exception as e:
    logger.error(f"JAX GPU initialization failed: {e}")
    logger.error("This may indicate GPU memory corruption from a previous job.")
    logger.error("Try requesting a different GPU node or contact cluster admin.")
    raise


class IdSumStat(pyabc.Sumstat):
    """Identity summary statistic for the pandas dataframe data."""

    def __call__(self, data: dict) -> np.ndarray:
        return data['data']


def main(abc_config,
         experiment_number,
         sampler='redis',
         redis_server=None,
         redis_port=6379):
    """Function to run the SMC-ABC algorithm using a trained Realcause model as the simulator.
    The dataset is specified by three arguments: dataset_name, dataset_identifier, and sample_size.

    Args:
        abc_config (dict): Dictionary containing the configuration parameters
                for the SMC-ABC algorithm.
        experiment_number (int): Identifier for the experiment to be run.
        sampler (str): Type of sampler to be used for the ABC algorithm, either 'singlecore' or 'redis'.
        redis_server (str): Hostname of the Redis server to be used for the ABC algorithm.
        redis_port (int): Port number for the Redis server.
    """
    dataset_name = abc_config['dataset_name']
    dataset_identifier = abc_config['dataset_identifier']
    sample_size = abc_config['sample_size']
    logger.info(f'Starting SMC-ABC algorithm for dataset: {dataset_name} with identifier: {dataset_identifier} and sample size: {sample_size}')
    
    # Load the observed dataset to be used as the reference dataset
    if dataset_name == 'lalonde':
        if dataset_identifier == 'psid1':
            d = rc_lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
        elif dataset_identifier == 'cps1':
            d = rc_lalonde.load_lalonde(obs_version='cps', data_format='pandas_single')
        else:
            raise ValueError(f"Dataset identifier {dataset_identifier} not implemented")
        d.drop(columns=['data_id'], inplace=True)
        outcome_col = 're78'
        treatment_col = 'treat'
        categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
        continuous_vars = ['age', 'education', 're75', 're74']
        # Sort the covariates columns to put the continous first, then the categorical
        covariates_col = continuous_vars + categorical_vars
        covariates_df = d[covariates_col].values   # This is the original covariates dataframe (not transformed)
    elif dataset_name == 'twins':
        if dataset_identifier == 'st':
            d = rc_twins.load_twins(data_format='pandas')
        covariates_col = d['w'].columns.tolist()
        covariates_df = d['w'].values
        observed_data = pd.concat([d['w'], d['t'], d['y']], axis=1)
        covariates_df = d['w'].values
        treatment_col = 'T'
        outcome_col = 'yf'
        categorical_vars = covariates_col
        continuous_vars = []
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    
    # Load the Realcause model from the specified path (before applying transformations)
    # We need to use the model's transforms to ensure scales match
    rc_model, _ = load_gen(saveroot=abc_config['realcause_model_path'])
    
    # Apply transformations using the model's transforms (if specified in config)
    # This ensures the observed data is normalized using the same parameters as the model
    if dataset_name in ['lalonde', 'twins']:
        if abc_config['transform'] == True:
            print(f'The covariates columns are: {covariates_col}')
            # The model's w_transform was created from training data
            # Transform the covariates using the model's transform
            transformed_w = rc_model.w_transform.transform(d[covariates_col].values)
            # Assign back to DataFrame columns
            for i, col in enumerate(covariates_col):
                d[col] = transformed_w[:, i]
            # Do the same for the outcome column
            transformed_y = rc_model.y_transform.transform(d[outcome_col].values.reshape(-1, 1))
            d[outcome_col] = transformed_y.flatten()
        # This column order is compatible with the FrugalFlows simulator as well
        d = d[[outcome_col, treatment_col] + covariates_col]
        observed_data = d
    
    # Numpy dictionary of the observed data
    observed = {
        'data': observed_data.values
    }
    # Sample_size observed
    observed_sample_size = observed_data.shape[0]
    
    # Training the FrugalFlows model
    # Convert the columns of the observed data to JAX compatible arrays
    X = jnp.array(observed_data[treatment_col].values, dtype=jnp.float64)[:, None]
    Y = jnp.array(observed_data[outcome_col].values, dtype=jnp.float64)[:, None]
    if len(categorical_vars) > 0:
        Z_disc = jnp.array(observed_data[categorical_vars].values, dtype=jnp.float64)
    else:
        Z_disc = None
    if len(continuous_vars) > 0:
        Z_cont = jnp.array(observed_data[continuous_vars].values, dtype=jnp.float64)
    else:
        Z_cont = None
    
    # Read in the hyperparameters for the trained models
    with open(abc_config['frugalflows_hp_file'], 'r', encoding='utf-8') as file:
        tuned_hyperparams = yaml.safe_load(file)

    # Set tuning parameters for Normalizing Flows
    max_patience = abc_config.get('max_patience', 200)    # Default is 200 TODO: Change after testing
    max_epochs = abc_config.get('max_epochs', 5000)    # Default is 5000 TODO: Change after testing
    tuned_hyperparams['hyperparameters']['max_patience'] = max_patience
    tuned_hyperparams['hyperparameters']['max_epochs'] = max_epochs

    # Train the frugal flow model with the hyperparameters specified in the config
    trained_ff_model = FrugalFlowModel(X=X,
                                       Y=Y,
                                       Z_disc=Z_disc,
                                       Z_cont=Z_cont,
                                       confounding_copula=None)
    # Set default causal model and corresponding arguments
    causal_model = abc_config.get('causal_model', 'location_translation')
    causal_model_args = None
    if causal_model == 'location_translation':
        trained_ff_model.train_benchmark_model(
            training_seed=jr.PRNGKey(abc_config['random_seed']),
            marginal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            frugal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            causal_model='location_translation',
            causal_model_args={
                'ate': tuned_hyperparams['ate'], **tuned_hyperparams['cm_hyperparameters']
            },
            prop_flow_hyperparam_dict=tuned_hyperparams['hyperparameters'],
        )
        # Print out the causal margin obtained after training the FF model
        learned_causal_margin = trained_ff_model.frugal_flow.bijection.bijections[-1].bijections[
            0].ate
        logger.info(f'Learned causal margin: {learned_causal_margin}')
    elif causal_model == 'gaussian':
        trained_ff_model.train_benchmark_model(
            training_seed=jr.PRNGKey(abc_config['random_seed']),
            marginal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            frugal_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            prop_flow_hyperparam_dict=tuned_hyperparams['hyperparameters'],
            causal_model='gaussian',
            causal_model_args={
                'ate': jnp.array([0.]),    # Starting values
                'const': 0.,
                'scale': 1
            })
        # Print out the causal margin obtained after training the FF model
        learned_causal_margin = trained_ff_model.frugal_flow.bijection.bijections[
            -1].bijection.bijections[0]
        logger.info(f'Learned causal margin: {learned_causal_margin.ate[0]}')
        logger.info(f'Learned causal margin: {learned_causal_margin.const}')
        logger.info(f'Learned scale: {learned_causal_margin.scale}')
        causal_model_args = {
            'const': learned_causal_margin.const, 'scale': learned_causal_margin.scale
        }
        
    # Define priors for the parameters 
    rc_prior_vars = abc_config['realcause_parameters']
    prior_distribution_name = {'normal': 'norm', 'uniform': 'uniform'}
    # Build a dictionary of prior distributions for each parameter
    rc_prior_dict = {}
    for param in rc_prior_vars:
        rc_prior_dict[param] = pyabc.RV(prior_distribution_name[abc_config['realcause_prior'][param]['distribution']],
                                     abc_config['realcause_prior'][param]['loc'],
                                     abc_config['realcause_prior'][param]['scale'])
    # Create joint prior distribution from all parameters
    rc_prior = pyabc.Distribution(**rc_prior_dict)
    logger.info(f'Realcause Prior distribution: {abc_config["realcause_prior"]}') 

    ff_prior_vars = abc_config['frugalflows_parameters']
    ff_prior_dict = {}
    for param in ff_prior_vars:
        ff_prior_dict[param] = pyabc.RV(prior_distribution_name[abc_config['frugalflows_prior'][param]['distribution']],
                                     abc_config['frugalflows_prior'][param]['loc'],
                                     abc_config['frugalflows_prior'][param]['scale'])
    ff_prior = pyabc.Distribution(**ff_prior_dict)
    logger.info(f'FrugalFlows Prior distribution: {abc_config["frugalflows_prior"]}') 
    
    # Construct a wrapper around the two simulator functions, so that we can pass in the parameters to both simulators
    def rc_simulator_pyabc(parameters):
        """Wrapper around the simulator function to be used by PyABC."""
        return rc_simulator.simulate_datasets(parameters=parameters,
                                           covariates_df=covariates_df,
                                           realcause_model=rc_model)
    
    def ff_simulator_pyabc(parameters):
        """Wrapper around the FrugalFlows simulator function to be used by PyABC."""
        return ff_simulator.simulate_datasets(parameters=parameters,
                                           sample_size=observed_sample_size,
                                           dataset_identifier=dataset_identifier,
                                           frugal_flow_model=trained_ff_model,
                                           causal_model=causal_model,
                                           causal_model_args=causal_model_args)

    # Define the distance metrics for the data (TODO: Add more distance functions later) 
    DISTANCE_PARAM = pyabc.SlicedWassersteinDistance(metric='sqeuclidean',
                                            p=2,
                                            sumstat=IdSumStat(),
                                            n_proj=50)    # Used to be 10
    observed_sum_stat = observed
    logger.info(f'Distance function: {abc_config["distance"]}')

    redis_sampler = pyabc.sampler.RedisEvalParallelSampler(host=redis_server, port=redis_port)
    if sampler == 'singlecore':
        sampler = pyabc.sampler.SingleCoreSampler()
    elif sampler == 'redis':
        sampler = redis_sampler
        
    # Initialize the ABC object
    abc = pyabc.ABCSMC(models=[rc_simulator_pyabc, ff_simulator_pyabc],
                       parameter_priors=[rc_prior, ff_prior],
                       distance_function=DISTANCE_PARAM,
                       population_size=30,
                       sampler=sampler,
                       eps=pyabc.MedianEpsilon())

    # Create the experiment name
    expt_name = f'{dataset_name}_{dataset_identifier}_{sample_size}_dist_{abc_config["distance"]}_expt_{experiment_number}'
    logger.info(f'Experiment name: {expt_name}')

    logger.info('Running ABC SMC')
    # Get a random number for the database name
    db_path = os.path.join('database', f'{expt_name}_{np.random.randint(100)}.db')
    logger.info(f'Using the following database: {db_path}')
    
    # Set initial model probabilities to 0.5 for both models (equal prior)
    # This ensures both models start with equal probability
    initial_model_probs = {0: 0.5, 1: 0.5}  # Model 0 (Realcause): 0.5, Model 1 (FrugalFlows): 0.5
    logger.info(f'Setting initial model probabilities: Realcause={initial_model_probs[0]}, FrugalFlows={initial_model_probs[1]}')
    
    abc.new(db='sqlite:///' + db_path, 
            observed_sum_stat=observed_sum_stat,
            model_probabilities=initial_model_probs)
    logger.info(f'Epsilon value: {abc_config["min_epsilon"]}')
    logger.info(f'-' * 50)
    logger.info(f'Configuration: {abc_config}')
    logger.info(f'-' * 50)
    history = abc.run(min_eps_diff=abc_config['min_epsilon'],
                      max_nr_populations=abc_config['max_iterations'])
    logger.info(
        'Stopping Criteria: Minimum Epsilon reached or Maximum Iterations reached, set to - ')
    logger.info(f'Minimum Epsilon: {abc_config["min_epsilon"]}')
    logger.info(f'Number of Generations: {abc_config["max_iterations"]}')
    logger.info('-' * 50)
    
    # Get model probabilities - this returns a DataFrame indexed by generation (t)
    # with columns corresponding to model IDs (0, 1, etc.)
    model_probs_df = history.get_model_probabilities()
    logger.info("Model probabilities over all generations:")
    logger.info(model_probs_df)
    
    # Verify initial model probabilities (generation 0) are 0.5 for both models
    if 0 in model_probs_df.index:
        initial_probs = model_probs_df.loc[0]
        prob_rc_init = float(initial_probs.get(0, 0.0))
        prob_ff_init = float(initial_probs.get(1, 0.0))
        logger.info(f"Initial model probabilities (generation 0): Realcause={prob_rc_init:.3f}, FrugalFlows={prob_ff_init:.3f}")
        if abs(prob_rc_init - 0.5) > 0.01 or abs(prob_ff_init - 0.5) > 0.01:
            logger.warning(f"Initial model probabilities are not exactly 0.5! This may indicate an issue.")
    else:
        logger.warning("Generation 0 not found in model probabilities DataFrame")
    
    # Verify initial model probabilities (generation 0) are 0.5 for both models
    if 0 in model_probs_df.index:
        initial_probs = model_probs_df.loc[0]
        prob_rc_init = float(initial_probs.get(0, 0.0))
        prob_ff_init = float(initial_probs.get(1, 0.0))
        logger.info(f"Initial model probabilities (generation 0): Realcause={prob_rc_init:.3f}, FrugalFlows={prob_ff_init:.3f}")
        if abs(prob_rc_init - 0.5) > 0.01 or abs(prob_ff_init - 0.5) > 0.01:
            logger.warning(f"Initial model probabilities are not exactly 0.5! This may indicate an issue.")
    else:
        logger.warning("Generation 0 not found in model probabilities DataFrame")
    
    # Get extended populations for both models
    extended_population_rc = history.get_population_extended(m=0, t='last', tidy=True)
    extended_population_ff = history.get_population_extended(m=1, t='last', tidy=True)
    
    logger.info(f"Realcause model (m=0) has {len(extended_population_rc)} particles")
    logger.info(f"FrugalFlows model (m=1) has {len(extended_population_ff)} particles")
    
    # Sample particles from each model according to their weights
    NUM_PARTICLES = 50
    
    # Extract model probabilities for the last generation
    # The DataFrame has columns 0, 1 (model IDs) and rows indexed by generation t
    # Note: If all particles come from one model, the DataFrame may only have one column
    if history.max_t in model_probs_df.index:
        last_gen_probs = model_probs_df.loc[history.max_t]
        # Use .get() with default 0.0 to handle missing model columns
        # (occurs when a model has no particles in that generation)
        prob_rc = float(last_gen_probs.get(0, 0.0))  # Model 0 (Realcause)
        prob_ff = float(last_gen_probs.get(1, 0.0))  # Model 1 (FrugalFlows)
        
        # Normalize probabilities if they don't sum to 1 (shouldn't happen, but safety check)
        total_prob = prob_rc + prob_ff
        if total_prob > 0:
            prob_rc = prob_rc / total_prob
            prob_ff = prob_ff / total_prob
        else:
            logger.warning("Both model probabilities are 0, using equal weights")
            prob_rc = 0.5
            prob_ff = 0.5
    else:
        logger.warning(f"Generation {history.max_t} not found in model probabilities, using equal weights")
        prob_rc = 0.5
        prob_ff = 0.5
    
    # Check if all particles come from the same model
    if prob_rc == 1.0 or prob_ff == 1.0:
        if prob_rc == 1.0:
            logger.warning("All particles come from Realcause model (prob_rc=1.0, prob_ff=0.0)")
        else:
            logger.warning("All particles come from FrugalFlows model (prob_rc=0.0, prob_ff=1.0)")
    
    # Ensure we have at least some particles from each model if both models have particles
    # If one model has 0 particles, we can't sample from it anyway
    num_particles_rc = int(np.round(NUM_PARTICLES * prob_rc))
    num_particles_ff = NUM_PARTICLES - num_particles_rc
    
    # Adjust if one model has no particles available
    if len(extended_population_rc) == 0:
        logger.warning("Realcause model has no particles available, adjusting particle counts")
        num_particles_rc = 0
        num_particles_ff = NUM_PARTICLES
    elif len(extended_population_ff) == 0:
        logger.warning("FrugalFlows model has no particles available, adjusting particle counts")
        num_particles_rc = NUM_PARTICLES
        num_particles_ff = 0
    
    logger.info(f"Sampling {num_particles_rc} particles from Realcause model (prob={prob_rc:.3f})")
    logger.info(f"Sampling {num_particles_ff} particles from FrugalFlows model (prob={prob_ff:.3f})")
    
    # Sample from each model's population
    sampled_particles_rc = extended_population_rc.sample(
        n=min(num_particles_rc, len(extended_population_rc)), 
        weights='w', 
        replace=False
    ) if num_particles_rc > 0 else pd.DataFrame()
    
    sampled_particles_ff = extended_population_ff.sample(
        n=min(num_particles_ff, len(extended_population_ff)), 
        weights='w', 
        replace=False
    ) if num_particles_ff > 0 else pd.DataFrame()
    
    # Add model identifier to each sample
    if not sampled_particles_rc.empty:
        sampled_particles_rc['model'] = 'realcause'
    if not sampled_particles_ff.empty:
        sampled_particles_ff['model'] = 'frugalflows'
    
    # Combine samples from both models
    sampled_particles = pd.concat([sampled_particles_rc, sampled_particles_ff], 
                                   ignore_index=True)
    
    logger.info(f"Total sampled particles: {len(sampled_particles)}")
    logger.info(f"Sampled particles by model:\n{sampled_particles['model'].value_counts()}")

    # Update experiment name based on ID
    expt_name = f'{expt_name}/{history.id}'
    logger.info(f'Experiment name: {expt_name}')

    ##### SAVE DATA #####
    # Create sub-folder within data
    os.makedirs(f'data/smc_abc/{expt_name}', exist_ok=True)

    # Save the observed data
    observed_data.to_csv(f'data/smc_abc/{expt_name}/observed.csv', index=False)

    NUM_SAMPLES = 50

    # Save the samples drawn from the posterior for BOTH models
    # Fit posterior distributions for each model (only if particles are available)
    posterior_rc = None
    posterior_ff = None
    
    if len(extended_population_rc) > 0:
        try:
            posterior_rc = pyabc.transition.MultivariateNormalTransition()
            posterior_rc.fit(*history.get_distribution(m=0, t=history.max_t))
            logger.info("Successfully fitted Realcause posterior")
        except Exception as e:
            logger.warning(f"Failed to fit Realcause posterior: {e}")
            posterior_rc = None
    else:
        logger.warning("Cannot fit Realcause posterior: no particles available")
    
    if len(extended_population_ff) > 0:
        try:
            posterior_ff = pyabc.transition.MultivariateNormalTransition()
            posterior_ff.fit(*history.get_distribution(m=1, t=history.max_t))
            logger.info("Successfully fitted FrugalFlows posterior")
        except Exception as e:
            logger.warning(f"Failed to fit FrugalFlows posterior: {e}")
            posterior_ff = None
    else:
        logger.warning("Cannot fit FrugalFlows posterior: no particles available")

    # Create a dataframe of the posterior and prior parameter samples for both models
    # Realcause parameters
    rc_post_var_names = ['rc_post_' + var for var in rc_prior_vars]
    rc_prior_var_names = ['rc_prior_' + var for var in rc_prior_vars]
    rc_prior_s = {x: [] for x in rc_prior_var_names}
    rc_post_s = {x: [] for x in rc_post_var_names}
    
    # FrugalFlows parameters
    ff_post_var_names = ['ff_post_' + var for var in ff_prior_vars]
    ff_prior_var_names = ['ff_prior_' + var for var in ff_prior_vars]
    ff_prior_s = {x: [] for x in ff_prior_var_names}
    ff_post_s = {x: [] for x in ff_post_var_names}

    # Calculate how many samples to draw from each model based on model probabilities
    # Only sample from models that have fitted posteriors
    if posterior_rc is None and posterior_ff is None:
        logger.error("No posteriors available for either model! Cannot sample.")
        num_samples_rc = 0
        num_samples_ff = 0
    elif posterior_rc is None:
        # Only FrugalFlows posterior available
        num_samples_rc = 0
        num_samples_ff = NUM_SAMPLES
    elif posterior_ff is None:
        # Only Realcause posterior available
        num_samples_rc = NUM_SAMPLES
        num_samples_ff = 0
    else:
        # Both posteriors available, use model probabilities
        num_samples_rc = int(np.round(NUM_SAMPLES * prob_rc))
        num_samples_ff = NUM_SAMPLES - num_samples_rc
    
    logger.info(f'Drawing {num_samples_rc} samples from Realcause posterior' + (' (posterior available)' if posterior_rc is not None else ' (no posterior available)'))
    logger.info(f'Drawing {num_samples_ff} samples from FrugalFlows posterior' + (' (posterior available)' if posterior_ff is not None else ' (no posterior available)'))

    sample_counter = 0
    
    # Sample from Realcause model
    if num_samples_rc > 0 and posterior_rc is not None:
        for i in range(num_samples_rc):
            # Prior
            prior_parameters = rc_prior.rvs()
            for var in rc_prior_var_names:
                rc_prior_s[var].append(prior_parameters[var[9:]])  # Remove 'rc_prior_' prefix
            logger.info(f'RC Prior parameters: {prior_parameters}')
            prior_samples = pd.DataFrame(rc_simulator_pyabc(prior_parameters)['data'])
            if dataset_name in ['lalonde', 'twins']:
                prior_samples.columns = [outcome_col, treatment_col] + covariates_col
            prior_samples.to_csv(f'data/smc_abc/{expt_name}/rc_prior_sample_{i}.csv', index=False)

            # Posterior
            posterior_parameters = posterior_rc.rvs()
            # Ensure parameters are within bounds
            for var in rc_post_var_names:
                var_name = var[8:]  # Remove 'rc_post_' prefix
                if var_name in ['deg_hetero', 'overlap']:
                    if posterior_parameters[var_name] < 0.0:
                        posterior_parameters[var_name] = 0.0
                    elif posterior_parameters[var_name] > 1.0:
                        posterior_parameters[var_name] = 1.0
            for var in rc_post_var_names:
                rc_post_s[var].append(posterior_parameters[var[8:]])
            logger.info(f'RC Posterior parameters: {posterior_parameters}')
            posterior_samples = pd.DataFrame(rc_simulator_pyabc(posterior_parameters)['data'])
            if dataset_name in ['lalonde', 'twins']:
                posterior_samples.columns = [outcome_col, treatment_col] + covariates_col
            posterior_samples.to_csv(f'data/smc_abc/{expt_name}/rc_posterior_sample_{i}.csv', index=False)
            sample_counter += 1
    else:
        logger.warning("Skipping Realcause sampling: no posterior available")

    # Sample from FrugalFlows model
    if num_samples_ff > 0 and posterior_ff is not None:
        for i in range(num_samples_ff):
            # Prior
            prior_parameters = ff_prior.rvs()
            for var in ff_prior_var_names:
                ff_prior_s[var].append(prior_parameters[var[9:]])  # Remove 'ff_prior_' prefix
            logger.info(f'FF Prior parameters: {prior_parameters}')
            prior_samples = pd.DataFrame(ff_simulator_pyabc(prior_parameters)['data'])
            if dataset_name in ['lalonde', 'twins']:
                prior_samples.columns = [outcome_col, treatment_col] + covariates_col
            prior_samples.to_csv(f'data/smc_abc/{expt_name}/ff_prior_sample_{i}.csv', index=False)

            # Posterior
            posterior_parameters = posterior_ff.rvs()
            # Ensure ATE parameters are within reasonable bounds if needed
            for var in ff_post_var_names:
                ff_post_s[var].append(posterior_parameters[var[8:]])  # Remove 'ff_post_' prefix
            logger.info(f'FF Posterior parameters: {posterior_parameters}')
            posterior_samples = pd.DataFrame(ff_simulator_pyabc(posterior_parameters)['data'])
            if dataset_name in ['lalonde', 'twins']:
                posterior_samples.columns = [outcome_col, treatment_col] + covariates_col
            posterior_samples.to_csv(f'data/smc_abc/{expt_name}/ff_posterior_sample_{i}.csv', index=False)
            sample_counter += 1
    else:
        logger.warning("Skipping FrugalFlows sampling: no posterior available")

    # Save the prior and posterior samples in a single DataFrame
    # Combine both models' parameters (only if samples exist)
    if len(rc_post_s) > 0 and len(rc_prior_s) > 0:
        rc_parameter_samples = pd.DataFrame(rc_post_s).join(pd.DataFrame(rc_prior_s))
        # Add model probabilities as columns
        rc_parameter_samples['model_prob_rc'] = prob_rc
        rc_parameter_samples['model_prob_ff'] = prob_ff
        rc_parameter_samples.to_csv(f'data/smc_abc/{expt_name}/rc_parameter_samples.csv', index=False)
        logger.info(f"Saved {len(rc_parameter_samples)} Realcause parameter samples")
    else:
        logger.warning("No Realcause parameter samples to save")
    
    if len(ff_post_s) > 0 and len(ff_prior_s) > 0:
        ff_parameter_samples = pd.DataFrame(ff_post_s).join(pd.DataFrame(ff_prior_s))
        # Add model probabilities as columns
        ff_parameter_samples['model_prob_rc'] = prob_rc
        ff_parameter_samples['model_prob_ff'] = prob_ff
        ff_parameter_samples.to_csv(f'data/smc_abc/{expt_name}/ff_parameter_samples.csv', index=False)
        logger.info(f"Saved {len(ff_parameter_samples)} FrugalFlows parameter samples")
    else:
        logger.warning("No FrugalFlows parameter samples to save")
    
    # Also save model probabilities over time
    model_probs_df.to_csv(f'data/smc_abc/{expt_name}/model_probabilities.csv', index=True)

    logger.info('Saved posterior and prior samples!')

    ##### PLOTTING #####
    # Create sub-folder within plots
    plotting_dir = f'plots/smc_abc/{expt_name}'
    os.makedirs(plotting_dir, exist_ok=True)

    # TODO: Add plotting code here
    # # For each particle in sampled_particles, plot the observed data and the sumstat_data
    # for particle_id in sampled_particles.index:
    #     print(f'Particle ID: {particle_id}')
    #     # Extracting the dataframe
    #     sumstat_data = sampled_particles.loc[particle_id]['sumstat_data_pd']
    #     plotting.marginal_t(obs=obs_data,
    #                         gen=sumstat_data,
    #                         figure_path=plotting_dir,
    #                         particle_id=particle_id,
    #                         treatment_col=treatment_col)
    #     plotting.marginal_y(obs=obs_data,
    #                         gen=sumstat_data,
    #                         figure_path=plotting_dir,
    #                         particle_id=particle_id,
    #                         outcome_col=outcome_col)
    #     plotting.conditional_y(obs=obs_data,
    #                            gen=sumstat_data,
    #                            figure_path=plotting_dir,
    #                            particle_id=particle_id,
    #                            treatment_col=treatment_col,
    #                            outcome_col=outcome_col)
    #     plotting.marginal_covariates(obs=obs_data,
    #                                  gen=sumstat_data,
    #                                  figure_path=plotting_dir,
    #                                  particle_id=particle_id,
    #                                  outcome_col=outcome_col,
    #                                  treatment_col=treatment_col)
    #     break    # For now, let us just plot one particle

    # if 'rho' not in prior_vars:
    #     # Plot the distribution of the causal effect parameter, note that because of the simulator
    #     # the minute we add unobserved confounding the CE will change.
    #     plt.figure()
    #     sns.boxplot(sampled_particles['par_ate'])
    #     plt.ylabel('Causal Effect Parameter')
    #     plt.title('Boxplot of Causal Effect Parameter')
    #     plt.savefig(f'plots/smc_abc/{expt_name}/ate_boxplot.png')
    # else:
    #     # Plot the distribution of the ate and unobserved confounding parameter as a joint plot
    #     sns.jointplot(x='par_ate', y='par_rho', data=sampled_particles, kind='kde')
    #     plt.savefig(f'plots/smc_abc/{expt_name}/ate_rho_jointplot.png')

    # Plot model probabilities over time
    pyabc.visualization.plot_model_probabilities(history)
    plt.savefig(f'plots/smc_abc/{expt_name}/model_probabilities.png')
    plt.close()
    
    # For creating the posterior plots for BOTH models
    for model_idx, (model_name, prior_vars, prior_config_key) in enumerate([
        ('realcause', rc_prior_vars, 'realcause_prior'),
        ('frugalflows', ff_prior_vars, 'frugalflows_prior')
    ]):
        # Check if this model has particles before plotting
        population = extended_population_rc if model_idx == 0 else extended_population_ff
        if len(population) == 0:
            logger.warning(f'Skipping plots for {model_name} model (m={model_idx}): no particles available')
            continue
            
        logger.info(f'Creating plots for {model_name} model (m={model_idx})')
        
        for t in [0, history.max_t]:
            try:
                if len(prior_vars) > 1:
                    # Grouped plots when there are more than a single parameter
                    pyabc.visualization.plot_kde_matrix_highlevel(
                        history, m=model_idx, t=t)
                    plt.savefig(f'plots/smc_abc/{expt_name}/{model_name}_kde_matrix_t{t}.png')
                    plt.close()
                else:
                    # Plot the marginal distribution when there are single parameters
                    pyabc.visualization.plot_kde_1d_highlevel(
                        history, x=prior_vars[0], m=model_idx, t=t, 
                        title=f'{model_name} - {prior_vars[0]} - t = {t}')
                    plt.savefig(f'plots/smc_abc/{expt_name}/{model_name}_parameterized_{prior_vars[0]}_t{t}.png')
                    plt.close()
            except Exception as e:
                logger.warning(f'Failed to create KDE plots for {model_name} at t={t}: {e}')

        # Credible intervals for this model
        try:
            pyabc.visualization.plot_credible_intervals(history,
                                                        levels=[0.95],
                                                        m=model_idx,
                                                        ts=[history.max_t],
                                                        show_kde_max_1d=True)
            plt.savefig(f'plots/smc_abc/{expt_name}/{model_name}_credible_intervals.png')
            plt.close()
        except Exception as e:
            logger.warning(f'Failed to create credible intervals plot for {model_name}: {e}')

        # Individual parameter KDE plots
        for var in prior_vars:
            try:
                pyabc.visualization.plot_kde_1d_highlevel(history,
                                                        x=var,
                                                        m=model_idx,
                                                        t=history.max_t,
                                                        xmin=abc_config[prior_config_key][var]['loc'] - 1.0,
                                                        xmax=abc_config[prior_config_key][var]['loc'] +
                                                        abc_config[prior_config_key][var]['scale'] + 1.0,
                                                        numx=100,
                                                        title=f'{model_name} - KDE of {var}')
                plt.savefig(f'plots/smc_abc/{expt_name}/{model_name}_kde_{var}.png')
                plt.close()
            except Exception as e:
                logger.warning(f'Failed to create KDE plot for {model_name} parameter {var}: {e}')

    # Create the credible interval plot for the last generation
    logger.info('Saved plots for individual posteriors, credible intervals, and model probabilities!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMC-ABC for the SBICE pipeline')
    parser.add_argument('--config',
                        type=str,
                        default='configs/experiments.yaml',
                        help='Path to configuration file')
    parser.add_argument('--expt_num', type=int, default=1, help='Experiment identifier number')
    parser.add_argument('--sampler',
                        type=str,
                        default='redis',
                        choices=['singlecore', 'redis'],
                        help='Type of sampler to be used for the ABC algorithm')
    parser.add_argument('--redis_server',
                        type=str,
                        default=None,
                        help='Hostname of the Redis server to be used for the ABC algorithm')
    parser.add_argument('--redis_port',
                        type=int,
                        default=6379,
                        help='Port number for the Redis server')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        expt_configs = yaml.safe_load(file)

    configuration = expt_configs[f'experiment_{args.expt_num}']
    # Logging
    logger.info(f'Configuration file: {args.config}')
    logger.info(f'Running experiment: {args.expt_num}')

    # Run the main function using the arguments in the config file
    if args.sampler == 'singlecore':
        main(configuration,
             experiment_number=args.expt_num,
             sampler=args.sampler)
    else:
        main(configuration,
             experiment_number=args.expt_num,
             sampler=args.sampler,
             redis_server=args.redis_server,
             redis_port=args.redis_port)