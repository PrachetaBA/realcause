import pyro
import pyro.distributions as dist


def linear_gaussian_model(z, t=None, y=None):
    sigma_t = pyro.sample("sigma_t", dist.Uniform(0., 10.))
    w_zt = pyro.sample('w_zt', dist.Normal(0., 10.))
    b_t = pyro.sample('b_t', dist.Normal(0., 10.))

    sigma_y = pyro.sample('sigma_y', dist.Uniform(0., 10.))
    w_ty = pyro.sample('w_ty', dist.Normal(0., 10.))
    w_zy = pyro.sample('w_zy', dist.Normal(0., 10.))
    b_y = pyro.sample('b_y', dist.Normal(0., 10.))

    with pyro.plate("data", z.shape[0]):
        t = pyro.sample("t_obs", dist.Normal(w_zt * z + b_t, sigma_t), obs=t)
        y = pyro.sample("y_obs", dist.Normal(w_ty * t + w_zy * z + b_y, sigma_y), obs=y)

    return t, y