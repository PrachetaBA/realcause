"""Script to perform SMC-ABC for the SBICE pipeline
using the Realcause simulator."""

# Import libraries
import argparse
import logging
import os 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
import yaml

from data_loaders import apo, lalonde
from sbi import simulator 


# Defing logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    if dataset_name in ['n_acic_4', 'jdk', 'postgres']:
        d = apo.get_apo_data(identifier=dataset_name, confound_func=dataset_identifier, 
                             data_format='pandas', return_ites=False, 
                             ret_counterfactual_outcomes=False,
                             sample_size=sample_size)
        # Get a pandas dataframe from the combination of the orig columns
        observed_data = pd.concat([d['w'], d['t'], d['y']], axis=1)
    elif dataset_name == 'lalonde':
        if dataset_identifier == 'psid1':
            d = lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
        elif dataset_identifier == 'cps1':
            d = lalonde.load_lalonde(obs_version='cps', data_format='pandas_single')
        # Get a pandas dataframe from the combination of the original columns
        observed_data = d
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    
    # Numpy dictionary of the observed data
    observed = {
        'data': observed_data.values
    }
    # Sample_size observed
    observed_sample_size = observed_data.shape[0]
    
    ##### return ####
    print(observed['data'])
    print(observed['data'].shape)
    print(observed_data.head())
    print(observed_data.shape)
    print(observed_sample_size)
    return 
    
    # Define priors for the parameters 
    prior_vars = abc_config['parameters']
    prior_distribution_name = {'normal': 'norm', 'uniform': 'uniform'}
    # Build a dictionary of prior distributions for each parameter
    prior_dict = {}
    for param in prior_vars:
        prior_dict[param] = pyabc.RV(prior_distribution_name[abc_config['prior'][param]['distribution']],
                                     abc_config['prior'][param]['loc'],
                                     abc_config['prior'][param]['scale'])
    # Create joint prior distribution from all parameters
    prior = pyabc.Distribution(**prior_dict)
    logger.info(f'Prior distribution: {abc_config["prior"]}') 
    
    # Construct a wrapper around the simulator function 
    def simulator_pyabc(parameters):
        """Wrapper around the simulator function to be used by PyABC."""
        return simulator.simulate_datasets(parameters=parameters,
                                           dataset_name=dataset_name,
                                           dataset_identifier=dataset_identifier,
                                           sample_size=sample_size,
                                           realcause_model_path=abc_config['realcause_model_path'])
        
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
    abc = pyabc.ABCSMC(models=simulator_pyabc,
                       parameter_priors=prior,
                       distance_function=DISTANCE_PARAM,
                       population_size=50,
                       sampler=sampler,
                       eps=pyabc.MedianEpsilon())

    # Create the experiment name
    expt_name = f'{dataset_name}_{dataset_identifier}_{sample_size}_dist_{abc_config["distance"]}_expt_{experiment_number}'
    logger.info(f'Experiment name: {expt_name}')

    logger.info('Running ABC SMC')
    # Get a random number for the database name
    db_path = os.path.join('database', f'{expt_name}_{np.random.randint(100)}.db')
    logger.info(f'Using the following database: {db_path}')
    abc.new(db='sqlite:///' + db_path, observed_sum_stat=observed_sum_stat)
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

    # As we have not implemented summary statistics, currently it is the identity
    # function for the data generated by the simulator.
    extended_population = history.get_population_extended(m=0, t='last', tidy=True)
    # Let us pick some samples from the posterior and plot them against the observed
    # data. We need to sample from the dataframe `ext_pop` according to the weights 'w'
    # and then simulate the data using the parameters.
    NUM_PARTICLES = 10
    sampled_particles = extended_population.sample(n=NUM_PARTICLES, weights='w', replace=False)

    # Update experiment name based on ID
    expt_name = f'{expt_name}/{history.id}'
    logger.info(f'Experiment name: {expt_name}')

    ##### SAVE DATA #####
    # Create sub-folder within data
    os.makedirs(f'data/smc_abc/{expt_name}', exist_ok=True)

    # Save the observed data
    observed_data.to_csv(f'data/smc_abc/{expt_name}/observed.csv', index=False)

    NUM_SAMPLES = 50

    # Save the samples drawn from the posterior
    posterior = pyabc.transition.MultivariateNormalTransition()
    posterior.fit(*history.get_distribution(m=0, t=history.max_t))

    # Create a dataframe of the posterior and prior parameter samples
    # Augment each of these variables with the prefix 'post_' and 'prior_'
    post_var_names = ['post_' + var for var in prior_vars]
    prior_var_names = ['prior_' + var for var in prior_vars]
    prior_s = {x: [] for x in prior_var_names}
    post_s = {x: [] for x in post_var_names}


    for i in range(NUM_SAMPLES):
        prior_parameters = prior.rvs()
        for var in prior_var_names:
            prior_s[var].append(prior_parameters[var[6:]])
        logger.info(f'Prior parameters: {prior_parameters}')
        prior_samples = pd.DataFrame(simulator_pyabc(prior_parameters)['data'])
        prior_samples.to_csv(f'data/smc_abc/{expt_name}/prior_sample_{i}.csv', index=False)

        posterior_parameters = posterior.rvs()
        for var in post_var_names:
            post_s[var].append(posterior_parameters[var[5:]])
        logger.info(f'Posterior parameters: {posterior_parameters}')
        posterior_samples = pd.DataFrame(simulator_pyabc(posterior_parameters)['data'])
        posterior_samples.to_csv(f'data/smc_abc/{expt_name}/posterior_sample_{i}.csv', index=False)

    # Save the prior and posterior samples in a single DataFrame with columns post_vars + prior_vars
    parameter_samples = pd.DataFrame(post_s)
    parameter_samples = parameter_samples.join(pd.DataFrame(prior_s))
    parameter_samples.to_csv(f'data/smc_abc/{expt_name}/parameter_samples.csv', index=False)

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

    # For creating the posterior plots
    for t in [0, history.max_t]:
        if len(prior_vars) > 1:
            # Grouped plots when there are more than a single parameter
            # df, w = history.get_distribution(m=0, t=t)
            # pyabc.visualization.plot_kde_matrix(df, w)    # Produces plots for each parameter
            pyabc.visualization.plot_kde_matrix_highlevel(
                history, m=0, t=t)    # Produces a single plot for all parameters
        else:
            # Plot the marginal distribution when there are single parameters
            pyabc.visualization.plot_kde_1d_highlevel(history, x=prior_vars[0], m=0, t=t, title=f't = {t}')

        plt.savefig(f'plots/smc_abc/{expt_name}/parameterized_{prior_vars[0]}_{t}.png')

    pyabc.visualization.plot_credible_intervals(history,
                                                levels=[0.95],
                                                m=0,
                                                ts=[history.max_t],
                                                show_kde_max_1d=True)
    plt.savefig(f'plots/smc_abc/{expt_name}/credible_intervals.png')


    for var in prior_vars:
        pyabc.visualization.plot_kde_1d_highlevel(history,
                                                x=var,
                                                m=0,
                                                t=history.max_t,
                                                xmin=abc_config['prior'][var]['loc'] - 1.0,
                                                xmax=abc_config['prior'][var]['loc'] +
                                                abc_config['prior'][var]['scale'] + 1.0,
                                                numx=100,
                                                title=f'KDE of {var}')
        plt.savefig(f'plots/smc_abc/{expt_name}/kde_{var}.png')

    # Create the credible interval plot for the last generation
    logger.info('Saved plots for individual posteriors and credible intervals!')
    
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