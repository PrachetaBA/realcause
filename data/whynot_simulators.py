import pandas as pd
from utils import to_pandas_df

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=
        "Using or importing the ABCs from 'collections' instead of from 'collections.abc'"
        " is deprecated since Python 3.3,and in 3.9 it will stop working")
    warnings.filterwarnings("ignore", message=
        "the imp module is deprecated in favour of importlib; see the module's"
        " documentation for alternative uses")

    from whynot.simulators import lalonde


def generate_lalonde_random_outcome(hidden_dim=64, alpha_scale=0.1, seed=0):
    # num_samples parameter is ignored by the simulator since the lalonde dataset is a fixed size
    (z, t, y), causal_effects = lalonde.experiments.run_lalonde(
        num_samples=None, hidden_dim=hidden_dim, alpha_scale=alpha_scale, seed=seed)
    df = to_pandas_df(z, t, y)
    return df, causal_effects