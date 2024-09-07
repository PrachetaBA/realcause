# pylint: disable=dangerous-default-value, line-too-long, invalid-name
"""Script contains the biasing function to subsampling APO data 
to create an observational dataset.

It follows the implementation of Gentzel et al. (2019) from 
the following repository: 
https://github.com/kakeith/rct_rejection_sampling/blob/main/causal_eval/sampling.py
"""
# Import libraries
import numpy as np
# import pandas as pd
from scipy.special import expit

SEED = 183
rng = np.random.default_rng(SEED)

def osrct_algorithm(data, confound_func_params={"para_form": "linear"}, treatment_col='T', confounding_vars=["C"]):
    """
    Gentzel et al. 2021 OSRCT algorithm

    Inputs:
        - data: pd.DataFrame with columns "C" (covariates), "T" (treatment), "Y" (outcome)
        - f_function: the type of function we want to pass in as the researcher-specified P(S|C)
        - rng: numpy random number generator (so we have the same seed throughout)
        - confound_func_params : Dictionary with the parameters for the function that creates p_SC
            from researcher_specified_function_for_confounding()

    Returns sampled data to induce confounding
    """
    # generate probabilities of S (selection) as a function of C (covariates)
    p_SC = researcher_specified_function_for_confounding(data,
                                                         confounding_vars=confounding_vars,
                                                         confound_func_params=confound_func_params)
    # generate bernoulli random variables
    bernoulli = rng.binomial(1, p_SC, len(data))
    # Sanity check #1
    assert(bernoulli.shape[0] == data.shape[0])
    # Assert that every item in bernoulli with even idx == odd idx,
    # This might not be the case because of the random sampling, so we
    # just repeat the first choice; if this is not set, then we will have
    # issues with the subsampling!
    first_sample = bernoulli[::2]
    bernoulli = np.repeat(first_sample, 2)
    assert all(bernoulli[::2] == bernoulli[1::2])

    # accept rows where (bernoulli == T)
    data_resampled = data.loc[data[treatment_col] == bernoulli, :]
    # Sanity check #2
    assert(data_resampled.shape[0] == data.shape[0] // 2)
    # return the resampled data
    data_resampled.reset_index()

    return data_resampled

def researcher_specified_function_for_confounding(
    data,
    confounding_vars=["C"],
    confound_func_params={"para_form": "linear", "intercept": -1, "weight": 2.5}
):
    """
    Biasing function to generate a relationship between the 
    biasing covariates and the treatment. 
    
    1. Linear function: p(S|C) = expit(I + W * C) (Single covariate) where I is the intercept and W is the weight.
    2. Linear function: p(S|C) = binary_piecewise (Single covariate) where zeta0 and zeta1 are the probabilities. (if covariate is binary)
    3. Nonlinear function: p(S|C) = expit(C1 * C1 + C2 * C2 + C3 * C3 + C1 * C1 * C2) (Three covariates)
    4. Nonlinear function: p(S|C) = expit(C1 * C1 + C2 * C2 + C3 * C3 + C4 * C4 + C5 * C5 + C1 * C1 * C2) (Five covariates)
    
    confound_func_params: 
        para_form: linear, binary_piecewise, nonlinear
        linear: 
            intercept: float
            weight: float
        binary_piecewise:
            zeta0: float
            zeta1: float
        nonlinear:
            C1: float
            C2: float
            C3: float
            C4: float (optional)
            C5: float (optional)
    """
    # check that the `confund_func_params` has what it needs
    assert confound_func_params.get("para_form") is not None
    if confound_func_params["para_form"] == "binary_piecewise":
        assert confound_func_params.get("zeta0") is not None
        assert confound_func_params.get("zeta1") is not None

    # Currently, this works only when there is a single covariate.
    # linear function
    if confound_func_params["para_form"] == "linear":       # Continous C
        p_TC = expit(confound_func_params['intercept'] + confound_func_params['weight'] * data[confounding_vars])
        p_TC = np.array(p_TC).reshape(1, -1)[0]
    # for binary C, specify a piecewise function with zeta0 and zeta1
    elif confound_func_params["para_form"] == "binary_piecewise":       # Binary C
        p_TC = np.array([confound_func_params["zeta1"] if c == 1 else confound_func_params["zeta0"] for c in data[confounding_vars]])
    # for nonlinear C, specify a nonlinear function with C1, C2, C3, C4, C5
    elif confound_func_params["para_form"] == "nonlinear":
        if len(confounding_vars) == 3:
            p_TC = expit(
                confound_func_params["C1"] * data[confounding_vars[0]]
                + confound_func_params["C2"] * data[confounding_vars[1]]
                + confound_func_params["C3"] * data[confounding_vars[2]]
                + confound_func_params["C1"] * data[confounding_vars[0]] * data[confounding_vars[1]]
            )
            # Original version by authors had hardcoded names for the biasing covariates
            # p_TC = expit(
            #     confound_func_params["C1"] * data["C1"]
            #     + confound_func_params["C2"] * data["C2"]
            #     + confound_func_params["C3"] * data["C3"]
            #     + confound_func_params["C1"] * data["C1"] * data["C2"]
            # )
        elif len(confounding_vars) == 5:
            p_TC = expit(
                confound_func_params["C1"] * data["C1"]
                + confound_func_params["C2"] * data["C2"]
                + confound_func_params["C3"] * data["C3"]
                + confound_func_params["C4"] * data["C4"]
                + confound_func_params["C5"] * data["C5"]
                + confound_func_params["C1"] * data["C1"] * data["C2"]
            )
        else:
            raise ValueError("Only 3 or 5 covariates are supported for the nonlinear function")
    return p_TC