"""Simulator (Realcause) for the SBICE pipeline.

In the case of Realcause, this loads the pre-trained model 
and generates the samples from it."""

# Import libraries
import numpy as np
from loading import load_gen
from data_loaders import apo

def simulate_datasets(parameters, 
                      dataset_name,
                      dataset_identifier,
                      sample_size,
                      realcause_model_path,
                      ):
    """Simulate the datasets from the specified realcause model
    using the parameters to tweak the model as required."""
    
    # Load the model from the specified path
    rc_model, _ = load_gen(saveroot=realcause_model_path)
    
    # We want to keep the same covariates as in the original dataset
    if dataset_name in ['n_acic_4', 'jdk', 'postgres']:
        d = apo.get_apo_data(identifier=dataset_name, confound_func=dataset_identifier, 
                         data_format='numpy', return_ites=False, 
                         ret_counterfactual_outcomes=False,
                         sample_size=sample_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    
    df_w, _, _ = d['w'], d['t'], d['y']
    
    te = None
    overlap = 1.0
    deg_hetero = 1.0
    if 'te' in parameters and parameters['te'] is not None:
        causal_effect = parameters['te'] 
    if 'overlap' in parameters and parameters['overlap'] is not None:
        overlap = parameters['overlap']
    if 'deg_hetero' in parameters and parameters['deg_hetero'] is not None:
        deg_hetero = parameters['deg_hetero']
    _, t, y = rc_model.sample(df_w,overlap=overlap,
                            causal_effect_scale=causal_effect,
                            deg_hetero=deg_hetero,
                            ret_counterfactuals=False)
    
    # Ensure t and y are column vectors
    t = t.reshape(-1, 1) if t.ndim == 1 else t
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    
    # Concatenate arrays horizontally
    generated_data = np.column_stack([df_w, t, y])
    
    return {
        'data': generated_data
    }
    
if __name__ == '__main__':
    # Testing purposes
    parameters = {
        'te': 1.0,
        'overlap': 0.1,
        'deg_hetero': 0.1
    }
    dataset_name = 'postgres'
    dataset_identifier = 'linear'
    sample_size = 3000
    realcause_model_path = f'results/{dataset_name}_{dataset_identifier}_{sample_size}/default'
    d = simulate_datasets(parameters, dataset_name, dataset_identifier, sample_size, realcause_model_path)
    print(d['data'].shape)