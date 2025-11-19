"""Simulator (Realcause) for the SBICE pipeline.

In the case of Realcause, this loads the pre-trained model 
and generates the samples from it."""

# Import libraries
import numpy as np

def simulate_datasets(parameters, 
                      covariates_df,
                      realcause_model,
                      ):
    """Simulate the datasets from the specified realcause model
    using the parameters to tweak the model as required."""
    
    causal_effect = None # By default, we do not set the causal effect
    overlap = 1.0
    deg_hetero = 1.0
    untransform = True # By default, we transform the data back to the original scale
    if 'te' in parameters and parameters['te'] is not None:
        causal_effect = parameters['te'] 
        untransform = False # If we are setting the causal effect, we do not want to transform the data back to the original scale, so that the causal effect scale is as specified in the parameters
    if 'overlap' in parameters and parameters['overlap'] is not None:
        overlap = parameters['overlap']
    if 'deg_hetero' in parameters and parameters['deg_hetero'] is not None:
        deg_hetero = parameters['deg_hetero']
    w, t, y = realcause_model.sample(covariates_df,overlap=overlap,
                            causal_effect_scale=causal_effect,
                            deg_hetero=deg_hetero,
                            ret_counterfactuals=False,
                            untransform=untransform)
    
    # Ensure t and y are column vectors
    t = t.reshape(-1, 1) if t.ndim == 1 else t
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    
    # Concatenate arrays horizontally
    generated_data = np.column_stack([y, t, w])
    
    return {
        'data': generated_data
    }
