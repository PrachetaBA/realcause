"""Script to create the Credence class.

The Credence class represents the autoencoder that is used to learn a generative
model of the data. The class is used to train the autoencoder and to generate
new data samples.

This script is rewritten from the source: https://github.com/harsh-parikh/credence-to-causal-estimation/tree/main/credence-v2
and has been updated to use pytorch-lightning version 2.3.1

Additionally, we want to be able to generate X | T or use the same covariates
as in the real data. (We assume that T is as-is in the real data, and do not provide an option
to learn a Bernoulli distribution for a new T.)

We also want to have a centralized definition of the parameters of the Credence model
which includes the following
1. treatment_effect_fn: A function that takes in X and returns the treatment effect
2. selection_bias_fn: A function that takes in X and T and returns the selection bias
3. effect_rigidity: A hyperparameter that controls the rigidity of the treatment effect
4. bias_rigidity: A hyperparameter that controls the rigidity of the selection bias
5. kld_rigidity: A hyperparameter that controls the rigidity of the KL divergence loss
"""

# Import libraries
import sys
import numpy as np
import pandas as pd
import torch
import lightning as pl
import tqdm
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Import the autoencoder model
from . import autoencoder
from . import autoencoder_uniform

# Define the logger
tb_logger = TensorBoardLogger('logs/tensorboard_logs', name='credence')


# Define a custom class for the progress bar
class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()    # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch,
                                   batch_idx)    # don't forget this :)
        percent = (batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')


# Define a progress bar
# pb_bar = LitProgressBar()
pb_bar = ProgressBar()

# Define a callback for model checkpointing
# Checkpoint for model X|T
checkpoint_callback_pre = ModelCheckpoint(
    monitor='val_loss',
    dirpath='logs/model_checkpoints',
    filename='treatment-epoch{epoch:02d}-val_loss{val_loss:.2f}',
    auto_insert_metric_name=False)
# Checkpoint for model Y|X,T
checkpoint_callback_post = ModelCheckpoint(
    monitor='val_loss',
    dirpath='logs/model_checkpoints',
    filename='outcome-epoch{epoch:02d}-val_loss{val_loss:.2f}',
    auto_insert_metric_name=False)


# Define the Credence class
class Credence:

    def __init__(
            self,
            data,    # dataframe
            post_treatment_var,    # list of post treatment variables
            treatment_var,    # list of treatment variable(s)
            categorical_var,    # list of variables which are categorical
            numerical_var,    # list of variables which are numerical
            var_bounds={},    # dictionary of bounds if certain variable is bounded
            generate_covariates=False,    # whether to generate covariates or use the same as in the real data
            use_uniform_encoder=False,
            use_gpu=True,
            treatment_effect_fn=lambda x: 0,    # function to generate treatment effect
            selection_bias_fn=lambda x,
        t: 0,    # function to generate selection bias
            effect_rigidity=0,    # hyperparameter for treatment effect
            bias_rigidity=0,    # hyperparameter for selection bias
            kld_rigidity=0,    # hyperparameter for KL divergence loss
    ):
        self.data_raw = data
        self.Ynames = post_treatment_var
        self.Tnames = treatment_var

        self.categorical_var = categorical_var
        self.numerical_var = numerical_var

        self.var_bounds = var_bounds
        self.use_uniform_encoder = use_uniform_encoder

        # preprocess data
        self.data_processed = self.preprocess(
            self.data_raw,
            self.Ynames,
            self.Tnames,
            self.categorical_var,
            self.numerical_var,
        )

        self.Xnames = [x for x in self.data_processed.columns if x not in self.Ynames + self.Tnames]

        self.generate_covariates = generate_covariates    # Flag to generate covariates or not

        # Initialize the parameters to be used for the generated data.
        self.treatment_effect_fn = treatment_effect_fn
        self.selection_bias_fn = selection_bias_fn
        self.effect_rigidity = effect_rigidity
        self.bias_rigidity = bias_rigidity
        self.kld_rigidity = kld_rigidity
        self.device = torch.device('cpu')
        if use_gpu:
            self.device = torch.device('cuda')

    # train generator; we use different hyperparameters
    # for the outcome and covariates model.
    def fit(self, c_hparams, o_hparams, max_epochs=500):
        """After the hyperparameters are all tuned, use this to pass the final configuration to the model."""
        # T is the treatment variable, and is a Bernoulli variable
        self.m_treat = self.data_processed[self.Tnames].mean()
        #####################################
        # We can also define an autoencoder for T from the data
        # in which uncomment the following code
        # self.m_treat = autoencoder.conVAE(
        #     df=self.data_processed,
        #     Xnames=[],
        #     Ynames=self.Tnames,
        #     cat_cols=self.categorical_var,
        #     var_bounds=self.var_bounds,
        #     config={'latent_dim':latent_dim,
        #             'hidden_dim':hidden_dim,
        #             'batch_size':batch_size,
        #             'lr':0.001},
        #     kld_rigidity=kld_rigidity,
        # )  # .to('cuda:0')
        # bar = pb.ProgressBar()
        # self.trainer_treat = pl.Trainer(
        #     max_epochs=max_epochs,
        #     callbacks=[bar],
        # )
        # self.trainer_treat.fit(
        #     self.m_treat, self.m_treat.train_loader, self.m_treat.val_loader
        # )
        #####################################

        # generator for X | T
        if self.generate_covariates:
            if self.use_uniform_encoder:
                self.m_pre = autoencoder_uniform.conVAE(
                    df=self.data_processed,
                    Xnames=self.Tnames,
                    Ynames=self.Xnames,
                    cat_cols=self.categorical_var,
                    var_bounds=self.var_bounds,
                    config={
                        'latent_dim': c_hparams['latent_dim'],
                        'hidden_dim': c_hparams['hidden_dim'],
                        'batch_size': c_hparams['batch_size'],
                        'lr': c_hparams['lr']
                    },
                    treatment_effect_fn=self.treatment_effect_fn,
                    selection_bias_fn=self.selection_bias_fn,
                    kld_rigidity=c_hparams['kld_rigidity'],    # self.kld_rigidity,
                    bias_rigidity=c_hparams['bias_rigidity'],
                    effect_rigidity=c_hparams['effect_rigidity']).to(self.device)
            else:
                self.m_pre = autoencoder.conVAE(
                    df=self.data_processed,
                    Xnames=self.Tnames,
                    Ynames=self.Xnames,
                    cat_cols=self.categorical_var,
                    var_bounds=self.var_bounds,
                    config={
                        'latent_dim': c_hparams['latent_dim'],
                        'hidden_dim': c_hparams['hidden_dim'],
                        'batch_size': c_hparams['batch_size'],
                        'lr': c_hparams['lr']
                    },
                    treatment_effect_fn=self.treatment_effect_fn,
                    selection_bias_fn=self.selection_bias_fn,
                    kld_rigidity=c_hparams['kld_rigidity'],    # self.kld_rigidity,
                    bias_rigidity=c_hparams['bias_rigidity'],
                    effect_rigidity=c_hparams['effect_rigidity']).to(self.device)

            self.trainer_pre = pl.Trainer(max_epochs=max_epochs,
                                          callbacks=[pb_bar, checkpoint_callback_pre],
                                          logger=tb_logger)
            self.trainer_pre.fit(self.m_pre, self.m_pre.train_loader, self.m_pre.val_loader)
        else:
            self.m_pre = None

        # generator for Y(1),Y(0) | X, T
        if self.use_uniform_encoder:
            self.m_post = autoencoder_uniform.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames + self.Tnames,
                Ynames=self.Ynames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config={
                    'latent_dim': o_hparams['latent_dim'],
                    'hidden_dim': o_hparams['hidden_dim'],
                    'batch_size': o_hparams['batch_size'],
                    'lr': o_hparams['lr']
                },
                potential_outcome=True,
                treatment_cols=self.Tnames,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                effect_rigidity=o_hparams['effect_rigidity'],
                bias_rigidity=o_hparams['bias_rigidity'],
                kld_rigidity=o_hparams['kld_rigidity']    #self.kld_rigidity,
            ).to(self.device)
        else:
            self.m_post = autoencoder.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames + self.Tnames,
                Ynames=self.Ynames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config={
                    'latent_dim': o_hparams['latent_dim'],
                    'hidden_dim': o_hparams['hidden_dim'],
                    'batch_size': o_hparams['batch_size'],
                    'lr': o_hparams['lr']
                },
                potential_outcome=True,
                treatment_cols=self.Tnames,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                effect_rigidity=o_hparams['effect_rigidity'],
                bias_rigidity=o_hparams['bias_rigidity'],
                kld_rigidity=o_hparams['kld_rigidity']    #self.kld_rigidity,
            ).to(self.device)
        self.trainer_post = pl.Trainer(max_epochs=max_epochs,
                                       callbacks=[pb_bar, checkpoint_callback_post],
                                       logger=tb_logger)
        self.trainer_post.fit(self.m_post, self.m_post.train_loader, self.m_post.val_loader)
        # returning trained generators
        return [self.m_treat, self.m_pre, self.m_post]

    def tune_covariates(self, hparams, max_epochs=100):
        """Function to fit the treatment model."""
        # generator for X | T
        #print('getting into tuning', self.use_uniform_encoder)
        if self.use_uniform_encoder:
            self.m_pre = autoencoder_uniform.conVAE(df=self.data_processed,
                                                    Xnames=self.Tnames,
                                                    Ynames=self.Xnames,
                                                    cat_cols=self.categorical_var,
                                                    var_bounds=self.var_bounds,
                                                    config=hparams,
                                                    treatment_effect_fn=self.treatment_effect_fn,
                                                    selection_bias_fn=self.selection_bias_fn,
                                                    kld_rigidity=hparams['kld_rigidity'],
                                                    bias_rigidity=hparams['bias_rigidity'],
                                                    effect_rigidity=hparams['effect_rigidity']).to(
                                                        self.device)
        else:
            self.m_pre = autoencoder.conVAE(df=self.data_processed,
                                            Xnames=self.Tnames,
                                            Ynames=self.Xnames,
                                            cat_cols=self.categorical_var,
                                            var_bounds=self.var_bounds,
                                            config=hparams,
                                            treatment_effect_fn=self.treatment_effect_fn,
                                            selection_bias_fn=self.selection_bias_fn,
                                            kld_rigidity=hparams['kld_rigidity'],
                                            bias_rigidity=hparams['bias_rigidity'],
                                            effect_rigidity=hparams['effect_rigidity']).to(
                                                self.device)
        #print('Start training...', max_epochs)
        self.trainer_pre = pl.Trainer(max_epochs=max_epochs,
                                      callbacks=[
                                          pb_bar,
                                          TuneReportCallback(
                                              {
                                                  'loss': 'val_loss', 'train_loss': 'train_loss'
                                              },
                                              on='validation_end')
                                      ])
        self.trainer_pre.fit(self.m_pre, self.m_pre.train_loader, self.m_pre.val_loader)

    def tune_outcome(self, hparams, max_epochs=100):
        """Function to tune the hyperparameters of the outcome model."""
        # generator for Y(1),Y(0) | X, T
        if self.use_uniform_encoder:
            self.m_post = autoencoder_uniform.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames + self.Tnames,
                Ynames=self.Ynames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config=hparams,
                potential_outcome=True,
                treatment_cols=self.Tnames,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                effect_rigidity=hparams['effect_rigidity'],
                bias_rigidity=hparams['bias_rigidity'],
                kld_rigidity=hparams['kld_rigidity'],
            ).to(self.device)
        else:
            self.m_post = autoencoder.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames + self.Tnames,
                Ynames=self.Ynames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config=hparams,
                potential_outcome=True,
                treatment_cols=self.Tnames,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                effect_rigidity=hparams['effect_rigidity'],
                bias_rigidity=hparams['bias_rigidity'],
                kld_rigidity=hparams['kld_rigidity'],
            ).to(self.device)

        self.trainer_post = pl.Trainer(
            max_epochs=max_epochs,
        # We do not need checkpointing for hyperparameter tuning
            callbacks=[
                pb_bar,
                TuneReportCallback({
                    'loss': 'val_loss', 'train_loss': 'train_loss'
                },
                                   on='validation_end')
            ])
        self.trainer_post.fit(self.m_post, self.m_post.train_loader, self.m_post.val_loader)

    # sample from generator
    def sample(self, num_samples=1000, data=None):
        # initializing latent variables from standard normal distribution
        if data is None:
            # Learn treatment model
            # pi_treat = (
            #     torch.zeros((num_samples, self.m_treat.latent_dim)),
            #     torch.zeros((num_samples, self.m_treat.latent_dim)),
            # )
            # Just sample from existing treatment variable
            T = torch.bernoulli(torch.ones((num_samples, 1)) * 0.5)

            # Model for T|X
            pi_pre = (
                torch.zeros((num_samples, self.m_pre.latent_dim)),
                torch.ones((num_samples, self.m_pre.latent_dim)),
            )
            # Model for Y|X,T
            pi_post = (
                torch.zeros((num_samples, self.m_post.latent_dim)),
                torch.ones((num_samples, self.m_post.latent_dim)),
            )

        else:
            num_samples = data.shape[0]
            T = torch.tensor(data[self.Tnames].values.astype(float)).float()
            Y = torch.tensor(data[self.Ynames].values.astype(float)).float()
            X = torch.tensor(data[self.Xnames].values.astype(float)).float()
            # pi_treat = self.m_treat.forward(T)
            if self.generate_covariates:
                pi_pre = self.m_pre.forward(X)
            pi_post = self.m_post.forward(Y)

        # sample from conVAE
        Tgen = T    #self.m_treat.sample(pi=pi_treat, x=torch.empty(size=(num_samples, 0)))
        if self.generate_covariates:
            Xgen = self.m_pre.sample(pi=pi_pre, x=Tgen)
        else:
            if data is None:
                raise ValueError('Data must be provided to generate covariates.')
            Xgen = X
        Ygen = self.m_post.sample(pi=pi_post, x=torch.cat((Xgen, Tgen), 1))
        Ygen_prime = self.m_post.sample(pi=pi_post, x=torch.cat((Xgen, 1 - Tgen), 1))

        # wrapping in a dataframe
        df = pd.DataFrame(Xgen.detach().numpy(), columns=self.Xnames)
        df_T = pd.DataFrame(Tgen.detach().numpy(), columns=self.Tnames)
        df_Y = pd.DataFrame(Ygen.detach().numpy(),
                            columns=['Y%d' % i for i in range(Ygen.detach().numpy().shape[1])])
        df_Y_prime = pd.DataFrame(
            Ygen_prime.detach().numpy(),
            columns=['Yprime%d' % i for i in range(Ygen.detach().numpy().shape[1])])
        df = df.join(df_T).join(df_Y)
        df_prime = df.join(df_Y_prime)
        return df, df_prime

    def preprocess(
        self, df, post_treatment_var, treatment_var, categorical_var, numerical_var
    ):    # this function preprocesses the categorical variables from objects to numerics

        # Create a copy to avoid modifying the original dataframe
        df_ = df.copy()

        # codifying categorical variables
        for col in categorical_var:
            if col in df_.columns:
                df_[col] = df_[col].astype('category').cat.codes

        # Numerical variables are already numeric, so no conversion needed
        # All columns (categorical, numerical, treatment, post_treatment) are preserved

        return df_
