
import json
import torch
import numpy as np
import pandas as pd
import torchcde
from datetime import datetime
from bdp.data.crypto.dataloaders import CryptoDataLoader, ADataLoader
from bdp.models_.sdes.neural_sde import Generator, Discriminator
from bdp.models_.sdes.neural_sde import gradient_penalty
from bdp.models_.utils.utils_ts import unfold_steps_ahead

import os
import sys
import torch
import numpy as np
from torch import nn
from pprint import pprint

from tqdm import tqdm
from torch.optim import Adam
from bdp import project_path
from bdp.utils.weird_functions import positive_parameter
from bdp.models_.basic_utils import set_debugging_scheme, generate_training_message

from bdp.data.m5.dataloaders import covariates_info
from bdp.models_.abstract_models import DeepBayesianModel
from bdp.models_.deep_architectures.tcn import TemporalConvNet
import torch.optim.swa_utils as swa_utils

from bdp.data.m5.dataloaders import M5_BLOCK
from torch.utils.data import DataLoader

from bdp.models_.deep_architectures.deep_nets import MLP
from torch.distributions import Poisson, Normal
from bdp.models_.m5 import embeddings
from bdp.models_.m5.embeddings import create_embeddings

from bdp.data.crypto.create_dataset_files import create_merged_dataframe

class CryptoSeq2Seq(DeepBayesianModel):
    """
    # Sequential stuff

    ['count','price','day','wday','month','year',
     'event_name_1','event_type_1','event_name_2',
     'event_type_2','snap']

     #Non sequential
     ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    """

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "crypto_seq2seq"

        DeepBayesianModel.__init__(self,
                                   model_name,
                                   model_dir=model_dir,
                                   data_loader=data_loader,
                                   **kwargs)

    def define_deep_models(self):
        self.ts = torch.linspace(0.,10.-1.,10)

        self.TCN = TemporalConvNet(self.dimension,
                                   self.num_channels,
                                   self.kernel_size,
                                   dropout=self.dropout)

        self.encode_to_hidden = nn.Linear(self.time_encoding,2*self.decoder_hidden_state)

        self.decoder_lstm = nn.LSTM(self.dimension,
                                    self.decoder_hidden_state,
                                    #dropout=self.dropout,
                                    batch_first=True)

        self.decode_output = nn.Linear(self.decoder_hidden_state, self.dimension)
        self.prediction_loss = torch.nn.MSELoss(reduction="mean")

    def to(self,device):
        self.ts = self.ts.to(device)
        self.TCN.to(device)
        self.encode_to_hidden.to(device)
        self.decoder_lstm.to(device)
        self.decode_output.to(device)

    @classmethod
    def get_parameters(cls):
        parameters = {"dropout": .4,
                      "dimension":3,# price, market cap, volume (pmv)
                      "decoder_hidden_state": 9,
                      "decoder_steps":6,
                      "decoder_rnn": "lstm",
                      "steps_ahead":4,
                      "kernel_size": 3,  # TCN values
                      "number_of_levels": 10,
                      "time_encoding": 8,
                      "model_path": os.path.join(project_path, 'results')}

        return parameters

    def set_parameters(self, **kwargs):
        self.time_encoding = kwargs.get("time_encoding")

        self.covariates_dimension = kwargs.get("covariates_dimension")
        # TCN parameters
        self.dimension = kwargs.get("dimension")
        self.number_of_levels = kwargs.get("number_of_levels")
        self.kernel_size = kwargs.get("kernel_size")
        self.num_channels = [self.time_encoding] * self.number_of_levels
        self.receptive_field = 2 ** len(self.num_channels)

        # Decoder parameters
        self.decoder_hidden_state = kwargs.get("decoder_hidden_state")
        self.dropout = kwargs.get("dropout")
        self.decoder_steps = kwargs.get("decoder_steps")
        self.steps_ahead = kwargs.get("steps_ahead")

    def update_parameters(self, dataloader, **kwargs):
        # kwargs.get("v_dynamic_recognition_parameters").update({"observable_dim": dataloader.vocabulary_dim})
        # json.dump(kwargs, open(self.parameter_path, "w"))
        databatch = next(dataloader.train.__iter__())
        series = databatch.pmv
        dimension = series.shape[2]
        kwargs.update({"dimension":dimension})
        kwargs.update({"steps_ahead":dataloader.steps_ahead})
        kwargs.update({"covariates_dimension":dataloader.covariates_dimension})

        return kwargs

    def loss(self, databatch, forward_results, dataloader, epoch):
        output, unfolded_series = forward_results
        loss = self.prediction_loss(output,unfolded_series[:,:,1:,:])
        return {"loss":loss}

    def metrics(self, databatch, forward_results, epoch, mode="evaluation", data_loader=None):
        return {}

    def encoder(self,databatch,use_case="train"):
        if use_case == "train":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]
            sequence_lenght_ = sequence_lenght - self.steps_ahead + 1
            pmv = databatch.pmv[:,:-(self.steps_ahead-1),:]
            pmv = pmv.permute(0,2,1)
            encoding = self.TCN(pmv).permute(0,2,1) #[batch_size,sequence_lenght_,time_encoding]
            encoder_ = encoding.reshape(batch_size * sequence_lenght_, -1)#[batch_size*sequence_lenght_,time_encoding]
            hidden_state = self.encode_to_hidden(encoder_)#[batch_size*sequence_lenght_,decoder_hidden_state*dimension]
            return hidden_state

        elif use_case == "prediction":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]
            sequence_lenght_ = sequence_lenght - self.steps_ahead + 1
            pmv = databatch.pmv[:,:,:]
            pmv = pmv.permute(0,2,1)
            encoding = self.TCN(pmv).permute(0,2,1) #[batch_size,sequence_lenght_,time_encoding]
            encoder_ = encoding.reshape(batch_size * sequence_lenght, -1)#[batch_size*sequence_lenght_,time_encoding]
            hidden_state = self.encode_to_hidden(encoder_)#[batch_size*sequence_lenght_,decoder_hidden_state*dimension]
            return hidden_state

    def decoder(self,hidden_state,past,batch_size,sequence_lenght_,use_case="train"):
        if use_case == "train":
            hidden_state = hidden_state.chunk(2, dim=-1)
            hidden_state = (hidden_state[0].unsqueeze(0).contiguous(),hidden_state[0].unsqueeze(0).contiguous())

            output, hidden_state = self.decoder_lstm(past, hidden_state)
            #output,[batch_size_sequence_lenght_,setps_ahead,hidden_dim]

            # reshape as unfolded time series
            output = output.reshape(batch_size * sequence_lenght_ * (self.steps_ahead - 1), self.decoder_hidden_state)
            output = self.decode_output(output)
            output = output.reshape(batch_size, sequence_lenght_, self.steps_ahead - 1, self.dimension)
            output = positive_parameter(output)

            return output, hidden_state

        elif use_case == "prediction":
            hidden_state = hidden_state.chunk(2, dim=-1)
            hidden_state = (hidden_state[0].unsqueeze(0).contiguous(),hidden_state[0].unsqueeze(0).contiguous())

            output = []
            out = past.reshape(batch_size*sequence_lenght_,self.dimension).unsqueeze(1)
            for time_index in range(self.steps_ahead):
                out, hidden_state =  self.decoder_lstm(out,hidden_state)
                out = self.decode_output(out)
                out = positive_parameter(out)
                output.append(out)
            output = torch.cat(output,dim=1)
            output = output.reshape(batch_size, sequence_lenght_, self.steps_ahead, self.dimension)

            return output, hidden_state

    def data_to_device(self,databatch):
        databatch = databatch._replace(ids=databatch.ids.to(self.device),
                                       max=databatch.max.to(self.device),
                                       pmv=databatch.pmv.to(self.device))

        return databatch

    def forward(self, databatch,use_case="train"):
        if use_case == "train":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]
            steps_ahead = self.steps_ahead
            sequence_lenght_ = sequence_lenght - steps_ahead + 1

            # we unfold the series so at each step on time we predict (steps_ahead)
            unfolded_series = series.unfold(dimension=1, size=steps_ahead, step=1).contiguous()
            unfolded_series = unfolded_series.reshape(batch_size * sequence_lenght_,
                                                      self.dimension,
                                                      steps_ahead)
            unfolded_series = unfolded_series.permute(0, 2, 1)

            hidden_state = self.encoder(databatch) #[batch_size*sequence_lenght_,time_encoding]
            output,hidden_state = self.decoder(hidden_state,unfolded_series[:, :-1, :],batch_size,sequence_lenght_)
            unfolded_series = unfolded_series.reshape(batch_size,sequence_lenght_,steps_ahead,self.dimension)
            return output, unfolded_series

        elif use_case == "prediction":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]

            # we unfold the series so at each step on time we predict (steps_ahead)
            hidden_state = self.encoder(databatch,use_case="prediction") #[batch_size*sequence_lenght,time_encoding]
            output,hidden_state = self.decoder(hidden_state,series[:, :, :],batch_size,
                                               sequence_lenght_=sequence_lenght,use_case="prediction")
            return output, series

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})

        return inference_parameters

    def initialize_inference(self, data_loader: ADataLoader, parameters=None, **inference_parameters) -> None:
        super().initialize_inference(data_loader=data_loader, parameters=parameters, **inference_parameters)
        regularizers = inference_parameters.get("regularizers")

        #self.schedulers = {}
        #for k, v in regularizers.items():
        #    if v is not None:
        #        lambda_0 = v["lambda_0"]
        #        percentage = v["percentage"]
        #        self.schedulers[k] = SigmoidScheduler(lambda_0=lambda_0,
        #                                              max_steps=self.expected_num_steps,
        #                                              decay_rate=50.,
        #                                              percentage_change=percentage)

        series = next(data_loader.train.__iter__()).pmv
        self.batch_size = series.shape[0]
        self.sequence_lenght = series.shape[1]
        self.sequence_lenght_ = self.sequence_lenght - self.steps_ahead + 1

class CryptoSeq2NeuralSDE(DeepBayesianModel):
    """
    # Sequential stuff

    """
    def __init__(self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "crypto_seq2nsde"

        DeepBayesianModel.__init__(self,
                                   model_name,
                                   model_dir=model_dir,
                                   data_loader=data_loader,
                                   **kwargs)

    def define_deep_models(self):
        ###############################
        # ENCODER-GENERATE
        ###############################
        self.ts = torch.linspace(0.,self.steps_ahead-1.,self.steps_ahead)

        self.TCN = TemporalConvNet(self.dimension,
                                   self.num_channels,
                                   self.kernel_size,
                                   dropout=self.dropout)
        self.encode_to_hidden = nn.Linear(self.time_encoding,self.conditional_hidden_state)

        self.generator = Generator(self.dimension, self.initial_noise_size, self.noise_size,
                                   self.hidden_size, self.mlp_size, self.num_layers,
                                   self.conditional_hidden_state,self.conditional_init,
                                   self.conditional)
        ###############################
        # DSICRIMINATOR
        ###############################
        self.discriminator = Discriminator(self.dimension, self.hidden_size, self.mlp_size, self.num_layers,
                                           self.conditional_hidden_state,self.conditional_init)

        ###############################
        # defining protected in  order to be able to use averaged
        ###############################
        self.generator_  = self.generator
        self.discriminator_ = self.discriminator

    def to(self,device):
        self.ts = self.ts.to(device)
        self.TCN.to(device)
        self.encode_to_hidden.to(device)
        self.generator.to(device)
        self.discriminator.to(device)

    @classmethod
    def get_parameters(cls):
        parameters = {"dropout": .4,
                      "dimension":3,# price, market cap, volume (pmv)
                      "initial_noise_size": 5,  # How many noise dimensions to sample at the start of the SDE.
                      "noise_size": 3,  # How many dimensions the Brownian motion has.
                      "hidden_size": 12,  # How big the hidden size of the generator SDE and the discriminator CDE are.
                      "mlp_size": 16,  # How big the layers in the various MLPs are.
                      "num_layers": 1,  # How many hidden layers to have in the various MLPs.
                      "steps_ahead":4,
                      "conditional": False,
                      "conditional_hidden_state": 9,
                      "conditional_init": True,
                      "kernel_size": 3,  # encoder TCN values
                      "number_of_levels": 10, #encoder
                      "time_encoding": 8, # encoder
                      "model_path": os.path.join(project_path, 'results')}

        return parameters

    def set_parameters(self, **kwargs):
        self.dropout = kwargs.get("dropout")
        self.time_encoding = kwargs.get("time_encoding")
        self.steps_ahead = kwargs.get("steps_ahead")

        # Encoder parameters
        self.covariates_dimension = kwargs.get("covariates_dimension")

        # TCN parameters
        self.dimension = kwargs.get("dimension")
        self.number_of_levels = kwargs.get("number_of_levels")
        self.kernel_size = kwargs.get("kernel_size")
        self.num_channels = [self.time_encoding] * self.number_of_levels
        self.receptive_field = 2 ** len(self.num_channels)

        # Conditional
        self.conditional_hidden_state = kwargs.get("conditional_hidden_state")
        self.conditional = kwargs.get("conditional")
        self.conditional_init = kwargs.get("conditional_init")

        # Decoder
        self.initial_noise_size = kwargs.get("initial_noise_size")
        self.noise_size = kwargs.get("noise_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.mlp_size = kwargs.get("mlp_size")
        self.num_layers = kwargs.get("num_layers")

    def update_parameters(self, dataloader, **kwargs):
        databatch = next(dataloader.train.__iter__())
        series = databatch.pmv
        dimension = series.shape[2]
        kwargs.update({"dimension":dimension})
        kwargs.update({"steps_ahead":dataloader.steps_ahead})
        kwargs.update({"covariates_dimension":dataloader.covariates_dimension})

        return kwargs

    def set_averaged(self):
        self.generator_ = self.averaged_generator.module
        self.discriminator_ = self.averaged_discriminator.module

    def unset_averaged(self):
        self.generator_  = self.generator
        self.discriminator_ = self.discriminator

    def evaluate_loss(self,dataloader,averaged=False):
        if averaged:
            self.set_averaged()

        with torch.no_grad():
            total_samples = 0
            total_loss = 0
            data_batch = next(dataloader.validate.__iter__())
            data_batch = self.data_to_device(data_batch)

            encoder_forward, generated_samples = self.generate(data_batch)
            generated_score = self.discriminator_(generated_samples, encoder_forward)
            real_score, real_samples = self.discriminate(data_batch, encoder_forward)

            loss = generated_score - real_score

            batch_size = real_samples.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

        if averaged:
            self.unset_averaged()

        return total_loss / total_samples

    def metrics(self, databatch, forward_results, epoch, mode="evaluation", data_loader=None):
        return {}

    def encoder(self,databatch):
        series = databatch.pmv
        batch_size = series.shape[0]
        sequence_lenght = series.shape[1]
        sequence_lenght_ = sequence_lenght - self.steps_ahead + 1
        pmv = databatch.pmv[:,:-(self.steps_ahead-1),:]
        pmv = pmv.permute(0,2,1)
        encoding = self.TCN(pmv).permute(0,2,1) #[batch_size,sequence_lenght_,time_encoding]
        encoder_ = encoding.reshape(batch_size * sequence_lenght_, -1)#[batch_size*sequence_lenght_,time_encoding]
        hidden_state = self.encode_to_hidden(encoder_)#[batch_size*sequence_lenght_,decoder_hidden_state*dimension]
        return hidden_state

    def data_to_device(self,databatch):
        databatch = databatch._replace(ids=databatch.ids.to(self.device),
                                       max=databatch.max.to(self.device),
                                       pmv=databatch.pmv.to(self.device))

        return databatch

    def forward(self, databatch):
        """
        series = databatch.pmv
        # we unfold the series so at each step on time we predict (steps_ahead)
        series = unfold_steps_ahead(series,self.steps_ahead)

        hidden_state = self.encoder(databatch) #[batch_size*sequence_lenght_,time_encoding]
        output,hidden_state = self.decoder(hidden_state,series[:, :-1, :])
        series = series.reshape(self.batch_size,self.sequence_lenght_,self.steps_ahead,self.dimension)
        """
        encoder_forward, generated_samples = self.generate(databatch,train=False)
        return encoder_forward, generated_samples

    def generate(self,data_batch,train=True):
        """

        :param data_batch:
        :return: encoder_forward, generated_samples
            generated_samples
        """
        if train:
            encoder_forward = self.encoder(data_batch)
            unfolded_series = unfold_steps_ahead(data_batch.pmv, self.steps_ahead)
            generated_samples = self.generator_(unfolded_series, self.ts, encoder_forward)

            return encoder_forward,generated_samples
        else:
            with torch.no_grad():
                encoder_forward = self.encoder(data_batch)
                unfolded_series = unfold_steps_ahead(data_batch.pmv, self.steps_ahead)
                generated_samples = self.generator_(unfolded_series, self.ts, encoder_forward)

                generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(self.ts)
                return encoder_forward, generated_samples

    def discriminate(self,data_batch,encoder_forward=None):
        if encoder_forward is None:
            encoder_forward = self.encoder(data_batch)
        unfolded_series = unfold_steps_ahead(data_batch.pmv, self.steps_ahead)
        new_ts = self.ts.unsqueeze(0).expand([unfolded_series.shape[0], self.steps_ahead]).unsqueeze(-1)
        real_samples = torch.cat([new_ts, unfolded_series], dim=-1)
        real_score = self.discriminator_(real_samples,encoder_forward)
        return real_score,real_samples

    def train_generator(self,data_batch):
        encoder_forward,generated_samples = self.generate(data_batch)
        generated_score = self.discriminator(generated_samples,encoder_forward)

        self.writer.add_scalar("train/generator_score", generated_score, self.number_of_generator_iterations)
        self.number_of_generator_iterations += 1

        generated_score.backward()
        self.generator_optimiser.step()
        self.generator_optimiser.zero_grad()
        self.discriminator_optimiser.zero_grad()

    def train_discriminator(self,data_batch):
        with torch.no_grad():
            encoder_forward,generated_samples = self.generate(data_batch)
        generated_score = self.discriminator(generated_samples,encoder_forward)
        real_score,real_samples = self.discriminate(data_batch,encoder_forward)

        penalty = gradient_penalty(generated_samples,real_samples,self.discriminator,encoder_forward)
        loss = generated_score - real_score

        self.writer.add_scalar("train/loss", loss, self.number_of_discriminator_iterations)
        self.number_of_discriminator_iterations +=1

        (self.gp_coeff * penalty - loss).backward()
        self.discriminator_optimiser.step()
        self.discriminator_optimiser.zero_grad()

    def inference(self,data_loader,**inference_parameters):
        self.initialize_inference(data_loader,**inference_parameters)
        trange = tqdm(range(self.number_of_epochs))
        with open(self.inference_path, "a+") as f:
            for epoch in trange:
                #train generator
                data_batch = next(data_loader.train.__iter__())
                data_batch = self.data_to_device(data_batch)
                self.train_generator(data_batch)
                #train discriminator
                for _ in range(self.ratio):
                    data_batch = next(data_loader.train.__iter__())
                    data_batch = self.data_to_device(data_batch)
                    self.train_discriminator(data_batch)
                total_unaveraged_loss = self.evaluate_loss(data_loader)

                # Stochastic weight averaging typically improves performance.
                if epoch > self.swa_step_start:
                    self.averaged_generator.update_parameters(self.generator)
                    self.averaged_discriminator.update_parameters(self.discriminator)
                    total_averaged_loss = self.evaluate_loss(data_loader,True)

                    trange.write(f"Step: {epoch:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "f"Loss (averaged): {total_averaged_loss:.4f}")
                    self.writer.add_scalar("validation/total_unaveraged_loss", total_unaveraged_loss, self.number_of_iterations_val)
                    self.writer.add_scalar("validation/total_averaged_loss", total_averaged_loss, self.number_of_iterations_val)

                    self.number_of_iterations_val+=1
                    if total_averaged_loss > self.best_loss:
                        model_evaluation = {"loss":total_averaged_loss,
                                            "unaveraged_loss":total_unaveraged_loss}

                        self.save_model()
                        self.best_loss = total_averaged_loss
                        self.best_eval = self.best_loss
                        self.time_f = datetime.now()
                        self.inference_results["epoch"] = epoch
                        self.inference_results["best_eval_time"] = (self.time_f - self.time_0).total_seconds()
                        self.inference_results["best_eval_criteria"] = self.best_eval
                        self.inference_results.update(model_evaluation)
                        json.dump(self.inference_results, f)
                        f.write("\n")
                        f.flush()
                else:
                    trange.write(f"Step: {epoch:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
                    self.writer.add_scalar("validation/total_unaveraged_loss",
                                           total_unaveraged_loss,
                                           self.number_of_iterations_val)

                    self.number_of_iterations_val+=1

            self.generator.load_state_dict(self.averaged_generator.module.state_dict())
            self.discriminator.load_state_dict(self.averaged_discriminator.module.state_dict())
            self.save_model(True)
        final_time = datetime.now()
        self.inference_results["final_time"] = (final_time - self.time_0).total_seconds()
        return self.inference_results

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None}})
        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        inference_parameters.update({"ratio": 5})  # How many discriminator training steps to take per generator training step.
        inference_parameters.update({"gp_coeff": 10})  # How much to regularise with gradient penalty.
        inference_parameters.update({"lr": 1e-3})  # Learning rate often needs careful tuning to the problem.
        inference_parameters.update({"batch_size": 1024})  # Batch size.
        inference_parameters.update({"steps": 6000})  # How many steps to train both generator and discriminator for.
        inference_parameters.update({"init_mult1": 3})  # Changing the initial parameter size can help.
        inference_parameters.update({"init_mult2": 0.5})  #
        inference_parameters.update({"weight_decay": 0.01})  # Weight decay.
        inference_parameters.update({"swa_step_start": 2})  # When to start using stochastic weight averaging.
        inference_parameters.update({"steps_per_print": 1})  # When to start using stochastic weight averaging.

        return inference_parameters

    def initialize_inference(self, data_loader: ADataLoader, parameters=None, **inference_parameters) -> None:
        super().initialize_inference(data_loader=data_loader, parameters=inference_parameters, **inference_parameters)
        self.regularizers = inference_parameters.get("regularizers")
        self.lr = inference_parameters.get("lr")
        self.weight_decay = inference_parameters.get("weight_decay")
        self.steps = inference_parameters.get("steps")
        self.init_mult1 = inference_parameters.get("init_mult1")
        self.init_mult2 = inference_parameters.get("init_mult2")
        self.ratio = inference_parameters.get("ratio")  # How many discriminator training steps to take per generator training step.
        self.gp_coeff = inference_parameters.get("gp_coeff")  # How much to regularise with gradient penalty.
        self.swa_step_start = inference_parameters.get("swa_step_start")
        self.steps_per_print = inference_parameters.get("steps_per_print")

        series = next(data_loader.train.__iter__()).pmv
        self.batch_size = series.shape[0]
        self.sequence_lenght = series.shape[1]
        self.sequence_lenght_ = self.sequence_lenght - self.steps_ahead + 1
        self.decoder_batch_size = self.batch_size * self.sequence_lenght_

        # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
        self.generator_optimiser = torch.optim.Adadelta(self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.discriminator_optimiser = torch.optim.Adadelta(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.averaged_generator = swa_utils.AveragedModel(self.generator)
        self.averaged_discriminator = swa_utils.AveragedModel(self.discriminator)

        # set start of iterations
        self.time_0 = datetime.now()
        self.best_loss = -np.inf
        self.number_of_discriminator_iterations = 0
        self.number_of_generator_iterations = 0
        self.number_of_iterations_val = 0

class PredictorsModelFactory(object):
    models: dict

    def __init__(self):
        self._models = {'crypto_seq2seq': CryptoSeq2Seq,
                        'crypto_seq2neural':CryptoSeq2NeuralSDE}

    def create(self, model_type: str, **kwargs):
        builder = self._models.get(model_type)
        if not builder:
            raise ValueError(f'Unknown recognition model {model_type}')
        return builder(**kwargs)

if __name__ == "__main__":
    from deep_fields import data_path
    s2s = True
    neuralSDE = False

    test_inference = False
    test_prediction = True
    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)

    kwargs = {"path_to_data":data_folder,
              "batch_size": 29,
              "steps_ahead":7,
              "date_string": date_string,
              "clean":"interpol",
              "span":"full"}

    data_loader = CryptoDataLoader('cpu', **kwargs)
    data_batch = next(data_loader.train.__iter__())

    ####################################################
    # SEQUENCE TO SEQUENCE (TCN to RNN)
    ####################################################
    if s2s:
        model_param = CryptoSeq2Seq.get_parameters()
        inference_param = CryptoSeq2Seq.get_inference_parameters()

        if test_inference:
            inference_param.update({"number_of_epochs":500})
            inference_param.update({"cuda":"cuda"})
            inference_param.update({"learning_rate": .00001})

            cs2s = CryptoSeq2Seq(**model_param,data_loader=data_loader)
            inference_results = cs2s.inference(data_loader, **inference_param)

        if test_prediction:
            model_dir = "C:/Users/cesar/Desktop/Projects/General/deep_random_fields/results/crypto_seq2seq_old/1622545528/"
            cs2s = CryptoSeq2Seq(model_dir=model_dir)
            forward_results = cs2s(data_batch,use_case="prediction")
            print(forward_results)

    ####################################################
    # SEQUENCE TO SEQUENCE (TCN to Neural)
    ####################################################
    if neuralSDE:
        model_param = CryptoSeq2NeuralSDE.get_parameters()
        inference_param = CryptoSeq2NeuralSDE.get_inference_parameters()

        if test_inference:
            inference_param.update({"number_of_epochs":1000})
            inference_param.update({"cuda":"cpu"})

            model_param.update({"conditional":True})
            cs2n = CryptoSeq2NeuralSDE(**model_param,data_loader=data_loader)
            inference_results = cs2n.inference(data_loader=data_loader,**inference_param)
        if test_prediction:
            pprint(inference_results)

    #-----------------------------------------------
    #model_dir = os.path.join(project_path, 'results', 'crypto_seq2seq', '1622548584')
    #CryptoSeq2Seq(model_dir=model_dir)

# -----------------------------------------------------------------------------------------------------------------------
# TRAFORMERS XL
# https://arxiv.org/pdf/1901.02860.pdf]

