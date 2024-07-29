from collections import OrderedDict


HP = OrderedDict(
    # dataset
    data=['acic'],
    dataroot=['base_datasets'],  # TODO: MODIFY THIS PATH LOCALLY
    # saveroot=['osapo_acic_4'],     # Specified in the arguments to train_generator_loop as 'exp_name'
    # train=[True],
    # eval=[True],
    # overwrite_reload=[''],

    # distribution of outcome (y)
    dist=['SigmoidFlow'],     # ['SigmoidFlow, FactorialGaussian'], see README for more options
    dist_args=[['ndim=10', 'base_distribution=normal']], # ['ndim=5', 'base_distribution=uniform']], # These arguments are when the model is SigmoidFlow
    atoms=[[]],
    # atoms=[[0.0], [0.0]],  # list of floats, or empty list

    # architecture
    n_hidden_layers=[1],
    dim_h=[16, 32, 64],
    activation=['ReLU'],

    # training params
    lr=[0.01, 0.05],
    batch_size=[32],
    num_epochs=[300],
    early_stop=[False],
    ignore_w=[False],
    grad_norm=['inf'],

    w_transform=['Standardize'],
    y_transform=['Normalize'],
    train_prop=[0.5],
    val_prop=[0.1],
    test_prop=[0.4],
    seed=[123],

    # evaluation
    num_univariate_tests=[30]
)