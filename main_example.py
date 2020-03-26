from pyro.infer.autoguide import AutoNormal
from DataGenModel import DataGenModel
from data.synthetic import generate_zty_linear_scalar_data
from models import linear_gaussian_model

df = generate_zty_linear_scalar_data(500)
gen_model = DataGenModel(df, linear_gaussian_model, AutoNormal, n_iters=2000)
gen_model.plot_ty_dists(n_samples_per_z=1, save_name='test')