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
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Import the autoencoder model
from . import autoencoder
from . import autoencoder_uniform

# Define the logger
tb_logger = TensorBoardLogger('logs/tensorboard_logs', name='modified_credence')


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
# Checkpoint for model T | X
checkpoint_callback_treat = ModelCheckpoint(
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
class MCredence:

    def __init__(
        self,
        data,    # dataframe
        post_treatment_var,    # list of post treatment variables
        treatment_var,    # list of treatment variable(s)
        categorical_var,    # list of variables which are categorical
        numerical_var,    # list of variables which are numerical
        var_bounds={},    # dictionary of bounds if certain variable is bounded
        treatment_effect_fn=lambda x: 0,    # function to generate treatment effect
        selection_bias_fn=lambda x,
        t: 0,    # function to generate selection bias
        effect_rigidity=0,    # hyperparameter for treatment effect
        bias_rigidity=0,    # hyperparameter for selection bias
        kld_rigidity=0.1,    # hyperparameter for KL divergence loss
        use_gpu=True,
        use_uniform_encoder=False,
    ):
        self.data_raw = data
        self.Ynames = post_treatment_var
        self.Tnames = treatment_var

        self.categorical_var = categorical_var
        self.numerical_var = numerical_var
        self.use_uniform_encoder = use_uniform_encoder

        self.var_bounds = var_bounds

        # preprocess data
        self.data_processed = self.preprocess(
            self.data_raw,
            self.Ynames,
            self.Tnames,
            self.categorical_var,
            self.numerical_var,
        )

        self.Xnames = [x for x in self.data_processed.columns if x not in self.Ynames + self.Tnames]

        #self.generate_covariates = generate_covariates # Flag to generate covariates or not
        #self.generate_treatment = generate_treatment # Flag to generate treatment or not

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
    def fit(self, t_hparams, o_hparams, max_epochs=250):
        """After the hyperparameters are all tuned, use this to pass the final configuration to the model."""
        # T is the treatment variable, and is a Bernoulli variable
        #self.m_treat = self.data_processed[self.Tnames].mean()
        #####################################
        # We can also define an autoencoder for T from the data
        # in which uncomment the following code
        if self.use_uniform_encoder:
            self.m_treat = autoencoder_uniform.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames,
                Ynames=self.Tnames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config=t_hparams,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                bias_rigidity=self.bias_rigidity,
                effect_rigidity=self.effect_rigidity,
                kld_rigidity=t_hparams['kld_rigidity']).float().to(self.device)
        else:
            self.m_treat = autoencoder.conVAE(df=self.data_processed,
                                              Xnames=self.Xnames,
                                              Ynames=self.Tnames,
                                              cat_cols=self.categorical_var,
                                              var_bounds=self.var_bounds,
                                              config=t_hparams,
                                              treatment_effect_fn=self.treatment_effect_fn,
                                              selection_bias_fn=self.selection_bias_fn,
                                              bias_rigidity=self.bias_rigidity,
                                              effect_rigidity=self.effect_rigidity,
                                              kld_rigidity=t_hparams['kld_rigidity']).float().to(
                                                  self.device)
        self.trainer_treat = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                pb_bar,
                TuneReportCallback({
                    'loss': 'val_loss', 'train_loss': 'train_loss'
                },
                                   on='validation_end'),    #checkpoint_callback_treat
            ],
            logger=tb_logger)
        self.trainer_treat.fit(self.m_treat, self.m_treat.train_loader, self.m_treat.val_loader)
        #####################################

        # generator for Y(1),Y(0) | X, T
        if self.use_uniform_encoder:
            self.m_post = autoencoder_uniform.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames + self.Tnames,
                Ynames=self.Ynames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config=o_hparams,
                potential_outcome=True,
                treatment_cols=self.Tnames,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                effect_rigidity=o_hparams['effect_rigidity'],
                bias_rigidity=o_hparams['bias_rigidity'],
                kld_rigidity=o_hparams['kld_rigidity'],
            ).to(self.device)
        else:
            self.m_post = autoencoder.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames + self.Tnames,
                Ynames=self.Ynames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config=o_hparams,
                potential_outcome=True,
                treatment_cols=self.Tnames,
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn,
                effect_rigidity=o_hparams['effect_rigidity'],
                bias_rigidity=o_hparams['bias_rigidity'],
                kld_rigidity=o_hparams['kld_rigidity'],
            ).to(self.device)

        self.trainer_post = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                pb_bar,
                TuneReportCallback({
                    'loss': 'val_loss', 'train_loss': 'train_loss'
                },
                                   on='validation_end'),    #checkpoint_callback_post
            ],
            logger=tb_logger)
        self.trainer_post.fit(self.m_post, self.m_post.train_loader, self.m_post.val_loader)

        # returning trained generators
        return [self.m_treat, self.m_post]

    def tune_treatment(self, hparams, max_epochs=100):
        """Function to fit the treatment model."""
        # generator for T | X
        if self.use_uniform_encoder:
            self.m_treat = autoencoder_uniform.conVAE(
                df=self.data_processed,
                Xnames=self.Xnames,
                Ynames=self.Tnames,
                cat_cols=self.categorical_var,
                var_bounds=self.var_bounds,
                config=hparams,
                kld_rigidity=hparams['kld_rigidity'],
                bias_rigidity=hparams['bias_rigidity'],
                effect_rigidity=hparams['effect_rigidity'],
                treatment_effect_fn=self.treatment_effect_fn,
                selection_bias_fn=self.selection_bias_fn).float().to(self.device)
        else:
            self.m_treat = autoencoder.conVAE(df=self.data_processed,
                                              Xnames=self.Xnames,
                                              Ynames=self.Tnames,
                                              cat_cols=self.categorical_var,
                                              var_bounds=self.var_bounds,
                                              config=hparams,
                                              kld_rigidity=hparams['kld_rigidity'],
                                              bias_rigidity=hparams['bias_rigidity'],
                                              effect_rigidity=hparams['effect_rigidity'],
                                              treatment_effect_fn=self.treatment_effect_fn,
                                              selection_bias_fn=self.selection_bias_fn).float().to(
                                                  self.device)

        self.trainer_treat = pl.Trainer(max_epochs=max_epochs,
                                        callbacks=[
                                            pb_bar,
                                            TuneReportCallback(
                                                {
                                                    'loss': 'val_loss', 'train_loss': 'train_loss'
                                                },
                                                on='validation_end')
                                        ])
        self.trainer_treat.fit(self.m_treat, self.m_treat.train_loader, self.m_treat.val_loader)

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
        num_samples = self.data_processed.shape[0]
        #print("no of samples: ", num_samples)
        X = torch.tensor(self.data_processed[self.Xnames].values).float()
        if data is None:
            # Learn treatment model
            # pi_treat = (
            #     torch.zeros((num_samples, self.m_treat.latent_dim)),
            #     torch.zeros((num_samples, self.m_treat.latent_dim)),
            # )
            # Just sample from existing treatment variable

            # Model for T|X
            pi_treat = (
                torch.zeros((num_samples, self.m_treat.latent_dim)),
                torch.ones((num_samples, self.m_treat.latent_dim)),
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
            #X = torch.tensor(data[self.Xnames].values.astype(float)).float()
            # pi_treat = self.m_treat.forward(T)
            #if self.generate_covariates:
            #pi_pre = self.m_pre.forward(X)
            #if self.generate_treatment:
            pi_treat = self.m_treat.forward(T)
            pi_post = self.m_post.forward(Y)

        # sample from conVAE
        #Tgen = T #self.m_treat.sample(pi=pi_treat, x=torch.empty(size=(num_samples, 0)))
        Tgen = self.m_treat.sample(pi=pi_treat, x=X)
        Xgen = X
        Ygen = self.m_post.sample(pi=pi_post, x=torch.cat((Xgen, Tgen), 1))
        Ygen_prime = self.m_post.sample(pi=pi_post, x=torch.cat((Xgen, 1 - Tgen), 1))

        # wrapping in a dataframe
        treat_var = self.Tnames[0]
        df = pd.DataFrame(Xgen.detach().numpy(), columns=self.Xnames)
        df_T = pd.DataFrame(Tgen.detach().numpy(), columns=self.Tnames)
        df_Y = pd.DataFrame(Ygen.detach().numpy(),
                            columns=['Y%d' % i for i in range(Ygen.detach().numpy().shape[1])])
        df_Y_prime = pd.DataFrame(
            Ygen_prime.detach().numpy(),
            columns=['Yprime%d' % i for i in range(Ygen.detach().numpy().shape[1])])
        df = df.join(df_T).join(df_Y)
        df_prime = df.join(df_Y_prime)
        df_prime['Y'] = (df_prime[treat_var] * df_prime['Y1']) + (
            (1 - df_prime[treat_var]) * df_prime['Y0'])
        df_prime['Y_cf'] = (df_prime[treat_var] * df_prime['Yprime1']) + (
            (1 - df_prime[treat_var]) * df_prime['Yprime0'])
        return df, df_prime

    def preprocess(
        self, df, post_treatment_var, treatment_var, categorical_var, numerical_var
    ):    # this function preprocesses the categorical variables from objects to numerics

        # codifying categorical variables
        df_cat = (df[categorical_var]).astype('category')
        for col in categorical_var:
            df_cat[col] = df_cat[col].cat.codes

        # codifying numeric variables
        df_num = df[numerical_var]

        # joining columns
        df_ = df_cat.join(df_num)

        return df_
