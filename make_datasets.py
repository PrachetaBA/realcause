"""Script that is used to generate realcause datasets using the models trained 
by the original authors Neal et al. 
"""

# Import libraries
import numpy as np
from numpy.testing import assert_approx_equal
from pathlib import Path
import time

from loading import load_from_folder

from data.lalonde import load_lalonde
from data.twins import load_twins
from consts import REALCAUSE_DATASETS_FOLDER, BASE_DATASETS_FOLDER, N_SAMPLE_SEEDS, N_AGG_SEEDS

SETTING = 10
FOLDER = Path(f'{REALCAUSE_DATASETS_FOLDER}/lalonde_psid_setting{SETTING}')
FOLDER.mkdir(parents=True, exist_ok=True)

psid_gen_model, args = load_from_folder(dataset='lalonde_psid1')
# cps_gen_model, args = load_from_folder(dataset='lalonde_cps1')
# twins_gen_model, args = load_from_folder(dataset='twins')

psid_w, psid_t, psid_y = load_lalonde(obs_version='psid', data_format='pandas')
# cps_w, cps_t, cps_y = load_lalonde(obs_version='cps', data_format='pandas')
# d_twins = load_twins(data_format='pandas')
# twins_w, twins_t, twins_y = d_twins['w'], d_twins['t'], d_twins['y']

gen_models = [psid_gen_model] #, [psid_gen_model, cps_gen_model, twins_gen_model]
w_dfs = [psid_w] #[psid_w, cps_w, twins_w]
names = ['lalonde_psid'] #['lalonde_psid', 'lalonde_cps', 'twins']

# Compute the naive ATE for PSID
psid_naive_ate = (psid_y[psid_t == 1].mean() - psid_y[psid_t == 0].mean())
print('ATE (Naive Biased Estimate) PSID:', psid_naive_ate)

# Compute the naive ATE estimate for CPS
# cps_naive_ate = (cps_y[cps_t == 1].mean() - cps_y[cps_t == 0].mean())
# print('ATE (Naive Biased Estimate) CPS:', cps_naive_ate)

# Compute the naive ATE estimate for Twins
# twins_naive_ate = (twins_y[twins_t == 1].mean() - twins_y[twins_t == 0].mean())
# print('ATE (Naive Biased Estimate) Twins:', twins_naive_ate)

dfs = []
print('N samples:', N_SAMPLE_SEEDS)
print('N seeds per sample:', N_AGG_SEEDS)
for gen_model, w_df, name in zip(gen_models, w_dfs, names):
    w_orig = w_df.to_numpy()

    dataset_start = time.time()
    print('Dataset:', name)
    ate_means = []
    for sample_i in range(N_SAMPLE_SEEDS):
        print('Sample:', sample_i)
        start_seed = sample_i * N_AGG_SEEDS
        end_seed = start_seed + N_AGG_SEEDS
        ates = []
        for seed in range(start_seed, end_seed):
            if SETTING == 1:
                # Setting 1: causal_effect_scale=1.0, deg_hetero=0.0
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                causal_effect_scale=1.0,  # Set to a value of 1.0 (scaled)
                                                deg_hetero=0.0,       # Set to a value of 0.0 (no heterogeneity)
                                                untransform=False) # Set to False so that the y values are left as is
            elif SETTING == 2:
                # Setting 2: causal_effect_scale=10.0
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    causal_effect_scale=10.0)  # Set to a value of 10.0 (scaled)
            elif SETTING == 3:
                # Setting 3: causal_effect_scale=-10.0
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    causal_effect_scale=-10.0)
            elif SETTING == 4:
                # Setting 4: causal_effect_scale=-10.0, untransform=False (Leave the y values as is)
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    causal_effect_scale=-10.0, untransform=False)
            elif SETTING == 5:
                # Setting 5: Now we will examine the effect of deg_hetero, without specifying any causal effect
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    deg_hetero=0.5)
            elif SETTING == 6:
                # Setting 6: Examine the effect of setting the degree of heterogeneity to 0.5, and having untransform=False
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    deg_hetero=0.5, untransform=False)
            elif SETTING == 7:
                # Setting 7: Examines the effect of the overlap parameter
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    overlap=0.1)            
            elif SETTING == 8:
                # Setting 8: Examines the effect of the overlap parameter with a more skewed distribution
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    overlap=0.9)
            elif SETTING == 9:
                # Setting 9: Set overlap to 0.0001
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    overlap=0.0001) 
            elif SETTING == 10:
                # Setting 10: Set the causal effect to -10.0 and the deg_hetero to 0, transform the y values back to the original scale
                w, t, (y0, y1) = gen_model.sample(w_orig, ret_counterfactuals=True, seed=seed,
                                                    causal_effect_scale=-10.0, deg_hetero=0.0, untransform=True)                       
            y = y0 * (1 - t) + y1 * t
            # w_errors = np.abs(w_orig - w)
            # assert w_errors.max() < 1e-2
            df = w_df
            df['t'] = t
            df['y'] = y
            df['y0'] = y0
            df['y1'] = y1
            df['ite'] = y1 - y0
            ate = (y1 - y0).mean()
            ates.append(ate)

        ate_means.append(np.mean(ates))
        df.to_csv(FOLDER / (name + '_sample{}.csv'.format(sample_i)), index=False)
        dfs.append(df)
        
    print('ATEs: ', ates)
    print('ATE mean mean (min-max) (std): {} ({} - {}) ({})'
          .format(np.mean(ate_means), np.min(ate_means), np.max(ate_means), np.std(ate_means)))
    print('Time elapsed: {} minutes'.format((time.time() - dataset_start) / 60))


# assert_approx_equal(psid_t.mean(), dfs[0]['t'].mean(), significant=1)
# assert_approx_equal(psid_y.mean(), dfs[0]['y'].mean(), significant=2)
# assert_approx_equal(cps_t.mean(), dfs[1]['t'].mean(), significant=1)
# assert_approx_equal(cps_y.mean(), dfs[1]['y'].mean(), significant=3)
# assert_approx_equal(twins_t.mean(), dfs[2]['t'].mean(), significant=2)
# assert_approx_equal(twins_y.mean(), dfs[2]['y'].mean(), significant=2)
