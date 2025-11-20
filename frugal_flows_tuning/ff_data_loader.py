"""Data loader specific to the FrugalFlows models, we need to also load the corresponding
realcause model so that we can normalize the data in the same way."""

# Import libraries
import jax
import jax.numpy as jnp
import pandas as pd
from loading import load_gen
from data_loaders import lalonde as rc_lalonde
from data_loaders import twins as rc_twins
# Some fixed settings
jnp.set_printoptions(precision=2)
jax.config.update('jax_enable_x64', True)

def load_data_ff(dataset_name, dataset_identifier=None, sample_size=None, rc_model_path=None):
    """Function to load the data defined by the identifier
    specifically for use for FrugalFlows models."""
    
    if dataset_name == 'lalonde':
        if dataset_identifier == 'psid1':
            d = rc_lalonde.load_lalonde(obs_version='psid', data_format='pandas_single')
            rc_model_path = 'results/GenModelCkpts/lalonde/psid1/save'
        elif dataset_identifier == 'cps1':
            d = rc_lalonde.load_lalonde(obs_version='cps', data_format='pandas_single')
            rc_model_path = 'results/GenModelCkpts/lalonde/cps1/dist_argsndim=32+base_distribution=normal-n_hidden_layers2-dim_h64-lr0.001-w_transformStandardize'
        else: 
            raise ValueError(f"Dataset identifier {dataset_identifier} not implemented")
        d.drop(columns=['data_id'], inplace=True)
        outcome_col = 're78'
        treatment_col = 'treat'
        categorical_vars = ['black', 'hispanic', 'married', 'nodegree']
        continuous_vars = ['age', 'education', 're75', 're74']
        covariates_col = continuous_vars + categorical_vars
        
        # Apply the transformations to the data as done in the Realcause model
        rc_model, _ = load_gen(rc_model_path)
        transformed_w = rc_model.w_transform.transform(d[covariates_col].values)
        # Assign back to DataFrame columns
        for i, col in enumerate(covariates_col):
            d[col] = transformed_w[:, i]
        # Do the same for the outcome column
        transformed_y = rc_model.y_transform.transform(d[outcome_col].values.reshape(-1, 1))
        d[outcome_col] = transformed_y.flatten()

        X = jnp.array(d[treatment_col].values, dtype=jnp.float64)[:, None]
        Y = jnp.array(d[outcome_col].values, dtype=jnp.float64)[:, None]
        if len(categorical_vars) > 0:
            Z_disc = jnp.array(d[categorical_vars].values, dtype=jnp.float64)
        else:
            Z_disc = None
        if len(continuous_vars) > 0:
            Z_cont = jnp.array(d[continuous_vars].values, dtype=jnp.float64)
        else:
            Z_cont = None
            
        # Sanity check on shape of data
        print(f'Treatment (X) shape: {X.shape}')
        print(f'Outcome (Y) shape: {Y.shape}')
        print(f'Discrete covariates (incl. T) (Z_disc) shape: {Z_disc.shape}')
        print(f'Continuous covariates (Z_cont) shape: {Z_cont.shape}')
        
    elif dataset_name == 'twins':
        if dataset_identifier == 'st':   # synthetic treatment according to Realcause paper
            d = rc_twins.load_twins(data_format='pandas')
            rc_model_path = 'results/GenModelCkpts/twins/twins/n_hidden_layers1-dim_h64-lr5e-05-w_transformNormalize'
        covariates_col = d['w'].columns.tolist()
        d = pd.concat([d['w'], d['t'], d['y']], axis=1)
        treatment_col = 'T'
        outcome_col = 'yf'
        categorical_vars = covariates_col
        continuous_vars = []
        
        # Apply the transformations to the data as done in the Realcause model
        rc_model, _ = load_gen(rc_model_path)
        transformed_w = rc_model.w_transform.transform(d[covariates_col].values)
        # Assign back to DataFrame columns
        for i, col in enumerate(covariates_col):
            d[col] = transformed_w[:, i]
        # Do the same for the outcome column
        transformed_y = rc_model.y_transform.transform(d[outcome_col].values.reshape(-1, 1))
        d[outcome_col] = transformed_y.flatten()
        
        X = jnp.array(d[treatment_col].values, dtype=jnp.float64)[:, None]
        Y = jnp.array(d[outcome_col].values, dtype=jnp.float64)[:, None]
        Z_cont = None
        Z_disc = jnp.array(d[categorical_vars].values, dtype=jnp.float64)
        
        # Sanity check on shape of data
        print(f'Treatment (X) shape: {X.shape}')
        print(f'Outcome (Y) shape: {Y.shape}')
        print(f'Discrete covariates (incl. T) (Z_disc) shape: {Z_disc.shape}')
        
    output = {'X': X, 'Y': Y, 'Z_disc': Z_disc, 'Z_cont': Z_cont}

    return output

if __name__ == '__main__':
    # Test the data loader
    dataset_name = 'twins'
    dataset_identifier = 'st'
    sample_size = None
    rc_model_path = 'results/GenModelCkpts/twins/twins/n_hidden_layers1-dim_h64-lr5e-05-w_transformNormalize'
    data = load_data_ff(dataset_name, dataset_identifier, sample_size, rc_model_path)
    print(data['X'])
    print(data['Y'])
    print(data['Z_disc'])
    print(data['X'].shape)
    print(data['Y'].shape)
    print(data['Z_disc'].shape)
