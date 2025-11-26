"""Script generates data from the tuned modified credence models.

Conda environment: rc-cred-sbi"""

# Import libraries
import argparse
import os
from tqdm import tqdm
import yaml

# Import data loader
import modified_credence as mcredence
import credence as credence
from credence_tuning.credence_data_loader import load_data_credence


class DataGenerator:
    """
    Class to generate data from the tuned Credence/Modified Credence models.
    """

    def __init__(self, gen_model, experiment_identifier, tuned_hparams, generation_setting=None):
        self.gen_model = gen_model
        self.tuned_hparams = tuned_hparams
        self.experiment_identifier = experiment_identifier
        self.generation_setting = generation_setting

    def generate_data(self):
        """Function to generate data from the tuned Credence/Modified Credence models."""
        # Extract the configuration parameters for the experiment
        with open('configs/credence_experiments.yaml', 'r', encoding='utf-8') as file:
            experiment_configs = yaml.safe_load(file)
        config = experiment_configs[f'expt_{self.experiment_identifier}']
        dataset_name = config['dataset_name']
        dataset_identifier = config['dataset_identifier']
        sample_size = config['sample_size']

        if dataset_name == 'lalonde':
            if dataset_identifier == 'psid1':
                rc_model_path = 'results/GenModelCkpts/lalonde/psid1/save'
            elif dataset_identifier == 'cps1':
                rc_model_path = 'results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize'
            else:
                raise ValueError(f'Dataset identifier {dataset_identifier} not implemented')
            source_dataset, source_dataset_info = load_data_credence(dataset_name, dataset_identifier, sample_size, rc_model_path)

            # Extract the value of generative model settings
            # We have a strong belief that there is no confounding bias
            if config['treatment_effect_val'] == 'true_ate':
                treatment_effect = source_dataset_info['true_ate']
            elif config['treatment_effect_val'] == 'incorrect_ate':
                treatment_effect = config['treatment_effect_fn']
            elif config['treatment_effect_val'] == 'flexible':
                treatment_effect = 0.0    # Default value is 0.0

            if config['confounding_bias_val'] == 'flexible':
                confounding_bias = 0.0    # Default value is 0.0
            elif config['confounding_bias_val'] == 'fixed_cb':
                confounding_bias = config['confounding_bias_fn']

            if self.gen_model == 'modified_credence':
                # Define the Modified Credence model
                modified_credence_model = mcredence.MCredence(
                    data=source_dataset,
                    post_treatment_var=[source_dataset_info['outcome_col']],
                    treatment_var=[source_dataset_info['treatment_col']],
                    categorical_var=source_dataset_info['categorical_vars'],
                    numerical_var=source_dataset_info['continuous_vars'],
                    treatment_effect_fn=lambda x: treatment_effect,
                    selection_bias_fn=lambda x, t: confounding_bias,
                    effect_rigidity=config['effect_rigidity'],
                    bias_rigidity=config['bias_rigidity'],
                    kld_rigidity=config['kld_rigidity'],
                    use_uniform_encoder=False,
                    use_gpu=False,
                )

                # Define the tuned_hyperparameters in a dictionary
                treatment_model_params = {
                    'latent_dim': self.tuned_hparams['t_latent_dim'],
                    'batch_size': self.tuned_hparams['t_batch_size'],
                    'hidden_dim': self.tuned_hparams['t_hidden_dim'],
                    'lr': self.tuned_hparams['t_lr'],
                    'kld_rigidity': self.tuned_hparams['t_kld_rigidity'],
                    'bias_rigidity': self.tuned_hparams['t_bias_rigidity'],
                    'effect_rigidity': self.tuned_hparams['t_effect_rigidity']
                }
                outcome_model_params = {
                    'latent_dim': self.tuned_hparams['latent_dim'],
                    'batch_size': self.tuned_hparams['batch_size'],
                    'hidden_dim': self.tuned_hparams['hidden_dim'],
                    'lr': self.tuned_hparams['lr'],
                    'kld_rigidity': self.tuned_hparams['kld_rigidity'],
                    'bias_rigidity': self.tuned_hparams['bias_rigidity'],
                    'effect_rigidity': self.tuned_hparams['effect_rigidity']
                }
                # Fit the modified credence model
                modified_credence_model.fit(treatment_model_params,
                                            outcome_model_params,
                                            max_epochs=self.tuned_hparams['max_epochs'])
                # Create the directory to store the generated data
                os.makedirs(f'{self.generation_setting["gen_data_dir"]}', exist_ok=True)
                # Generate the data
                for itr in tqdm(range(self.generation_setting['num_samples'])):
                    gen_data_fname = f'dataset_{itr}'
                    df_gen, df_gen_prime = modified_credence_model.sample(source_dataset.shape[0],
                                                                          data=source_dataset)
                    df_gen.to_csv(f'{self.generation_setting["gen_data_dir"]}/{gen_data_fname}.csv',
                                  index=False)
                    df_gen_prime.to_csv(
                        f'{self.generation_setting["gen_data_dir"]}/{gen_data_fname}_prime.csv',
                        index=False)

            elif self.gen_model == 'credence':
                # Define the Credence model
                credence_model = credence.Credence(
                    data=source_dataset,
                    post_treatment_var=[source_dataset_info['outcome_col']],
                    treatment_var=[source_dataset_info['treatment_col']],
                    categorical_var=source_dataset_info['categorical_vars'],
                    numerical_var=source_dataset_info['continuous_vars'],
                    treatment_effect_fn=lambda x: treatment_effect,
                    selection_bias_fn=lambda x, t: confounding_bias,
                    effect_rigidity=config['effect_rigidity'],
                    bias_rigidity=config['bias_rigidity'],
                    kld_rigidity=config['kld_rigidity'],
                    use_uniform_encoder=False,
                    generate_covariates=
                    True,    # Has to be explicitly set to be faithful to the credence model
                    use_gpu=False)

                # Define the tuned_hyperparameters in a dictionary
                covariate_model_params = {
                    'latent_dim': self.tuned_hparams['c_latent_dim'],
                    'batch_size': self.tuned_hparams['c_batch_size'],
                    'hidden_dim': self.tuned_hparams['c_hidden_dim'],
                    'lr': self.tuned_hparams['c_lr'],
                    'kld_rigidity': self.tuned_hparams['c_kld_rigidity'],
                    'bias_rigidity': self.tuned_hparams['c_bias_rigidity'],
                    'effect_rigidity': self.tuned_hparams['c_effect_rigidity']
                }
                outcome_model_params = {
                    'latent_dim': self.tuned_hparams['latent_dim'],
                    'batch_size': self.tuned_hparams['batch_size'],
                    'hidden_dim': self.tuned_hparams['hidden_dim'],
                    'lr': self.tuned_hparams['lr'],
                    'kld_rigidity': self.tuned_hparams['kld_rigidity'],
                    'bias_rigidity': self.tuned_hparams['bias_rigidity'],
                    'effect_rigidity': self.tuned_hparams['effect_rigidity']
                }
                # Fit the Credence model
                credence_model.fit(covariate_model_params,
                                   outcome_model_params,
                                   max_epochs=self.tuned_hparams['max_epochs'])
                # Create the directory to store the generated data
                os.makedirs(f'{self.generation_setting["gen_data_dir"]}', exist_ok=True)
                # Generate the data
                for itr in tqdm(range(self.generation_setting['num_samples'])):
                    gen_data_fname = f'dataset_{itr}'
                    df_gen, df_gen_prime = credence_model.sample(source_dataset.shape[0],
                                                                 data=source_dataset)
                    df_gen.to_csv(f'{self.generation_setting["gen_data_dir"]}/{gen_data_fname}.csv',
                                  index=False)
                    df_gen_prime.to_csv(
                        f'{self.generation_setting["gen_data_dir"]}/{gen_data_fname}_prime.csv',
                        index=False)
            else:
                raise ValueError(f'Generation model {self.gen_model} not implemented')

        else:
            raise ValueError(f'Dataset {dataset_name} not implemented')


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', type=str, required=True)
    parser.add_argument('--experiment_identifier', type=str, required=True)
    args = parser.parse_args()

    if args.gen_model == 'credence':
        gen_data_dir = f'data/generated_datasets/credence/expt_{args.experiment_identifier}'
        hparam_file = f'results/credence_models/expt_{args.experiment_identifier}.yaml'
    elif args.gen_model == 'modified_credence':
        gen_data_dir = f'data/generated_datasets/modified_credence/expt_{args.experiment_identifier}'
        hparam_file = f'results/mcredence_models/expt_{args.experiment_identifier}.yaml'
    else:
        raise ValueError(f'Generation model {args.gen_model} not implemented')

    generation_setting = {'num_samples': 50, 'gen_data_dir': gen_data_dir}

    with open(hparam_file, 'r', encoding='utf-8') as file:
        tuned_hparams = yaml.safe_load(file)

    # Generate the data
    data_gen = DataGenerator(gen_model=args.gen_model,
                             experiment_identifier=args.experiment_identifier,
                             tuned_hparams=tuned_hparams,
                             generation_setting=generation_setting)
    data_gen.generate_data()
