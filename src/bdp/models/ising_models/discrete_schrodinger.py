import os
import json
import torch
from torch import nn

from pprint import pprint
from datetime import datetime
from deep_fields.models.deep_architectures.deep_nets import MLP

import tqdm
from abc import ABC, abstractmethod

import time
from torch.nn import Linear
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Poisson, Exponential,Bernoulli
from deep_fields.models.deep_architectures.deep_nets import MLP
from deep_fields.models.utils.basic_setups import create_dir_and_writer


class bernoulli_spins():

    def __init__(self,p):
        self.p = p
        self.bernoulli_distribution = Bernoulli(p)

    def sample(self,sample_shape):
        sample_one_zeros = self.bernoulli_distribution.sample(sample_shape=sample_shape)
        sample_one_zeros[torch.where(sample_one_zeros == 0)] = -1.
        return sample_one_zeros

class discrete_schrodinger_bridge(ABC,nn.Module):
    """
    Here we follow
    """
    def __init__(self,results_path=None,**kwargs):
        if results_path is None:
            self.input_parameters = kwargs
            experiments_class = kwargs.get("experiments_class")
            # META DATA
            model_identifier = str(int(time.time()))
            self.writer, self.results_path, self.best_model_path = create_dir_and_writer(model_name="discrete_schrodinger",
                                                                                         experiments_class="sinkhorn",
                                                                                         model_identifier="{0}".format(model_identifier),
                                                                                         delete=True)
            parameters_path = os.path.join(self.results_path, "parameters.json")
            super(discrete_schrodinger_bridge, self).__init__()
            json.dump(kwargs, open(parameters_path, "w"))
            self.set_parameters(**kwargs)
            self.define_deep_models()
        else:
            super(discrete_schrodinger_bridge, self).__init__()
            self.results_path = results_path
            results = torch.load(results_path)
            self.best_model_path = results_path
            kwargs = results.get("input_parameters")
            self.set_parameters(**kwargs)
            self.define_deep_models()
            self.PHI_0 = results.get("PHI_0")
            self.PHI_1 = results.get("PHI_1")

    @classmethod
    def get_parameters(cls):
        kwargs = {
            "T": 3.,
            "tau": 0.01,
            "time_embedding_dim": 10,
            "discrete_time_networks":False,
            "number_of_paths": 500,
            "max_number_of_time_steps": 20000,
            "number_of_estimate_iterations": 3000,
            "estimate_iterations_lr":1e-3,
            "epsilon_threshold": 1e-3,
            "clip_max_norm":None,
            "number_of_sinkhorn_iterations": 30}

        phi_parameters = {
            "input_dim": 1,
            "output_dim": 1,
            "layers_dim": [50, 50],
            "output_transformation": None,
            "dropout": 0.4
        }
        kwargs["phi_parameters"] = phi_parameters
        return kwargs

    def set_parameters(self,**kwargs):
        self.T = kwargs.get("T")
        self.tau = kwargs.get("tau")
        self.time_grid = torch.arange(0., self.T + self.tau, self.tau)
        self.number_of_time_steps = len(self.time_grid)-1

        self.number_of_paths = kwargs.get("number_of_paths")
        self.epsilon_threshold = kwargs.get("epsilon_threshold")
        self.time_embedding_dim = kwargs.get("time_embedding_dim")
        self.max_number_of_time_steps = kwargs.get("max_number_of_time_steps")
        self.number_of_estimate_iterations = kwargs.get("number_of_estimate_iterations")
        self.estimate_iterations_lr = kwargs.get("estimate_iterations_lr")
        self.clip_max_norm = kwargs.get("clip_max_norm")
        self.number_of_sinkhorn_iterations = kwargs.get("number_of_sinkhorn_iterations")
        self.phi_parameters = kwargs.get("phi_parameters")

        self.discrete_time_networks = kwargs.get("discrete_time_networks")

    @abstractmethod
    def define_deep_models(self):
        """
        :return:
        """
        if self.discrete_time_networks:
            self.PHI_0 = []
            self.PHI_1 = []
            for tau in range(self.number_of_time_steps):
                self.PHI_0.append(MLP(**self.phi_parameters))
                self.PHI_1.append(MLP(**self.phi_parameters))
        else:
            self.PHI_0 = MLP(**self.phi_parameters)
            self.PHI_1 = MLP(**self.phi_parameters)

    @abstractmethod
    def reference_process(self,paths):
        return None

    @abstractmethod
    def parametric_process(self,paths):
        return None

    @abstractmethod
    def symmetric_function_for_one_spin_flip(self, new_states, old_states, energy=False):
        return None

    @abstractmethod
    def estimator_loss_continuous_time(self,states,states_n_time,sinkhorn_index):
        return None

    @abstractmethod
    def estimator_loss_discrete_time(self,states,states_n_time,sinkhorn_index):
        return None

    @abstractmethod
    def select_random_path_position_continuous(self):
        """
        Here one should select a random position in the path
        and then encode such time, and appended to the states
        in the path, so one can evaluate the neural network
        in charge of the estimation

        :return:
        """
        return None

    @abstractmethod
    def repeat_and_new(self):
        return None

    @abstractmethod
    def initialize_inference(self):
        self.sinkhorn_index = 0
        self.max_steps_grid = self.time_grid.shape[0]

    #=============================================
    # BRIDGES
    #=============================================
    def forward(self):
        return None

    def sample_start_of_path(self,sinkhorn_index=0,training=True,forward=False):
        """
        here we handle the ordering of the time grid as well as the beginning of the
        path from the distribution sample

        this is according to the sinkhorn index

        :param sinkhorn_index:
        :param training:
        :param forward:
        :return:
        """
        if training:
            if sinkhorn_index % 2 == 0:
                paths = self.initial_distribution.sample([self.number_of_paths]) # DATA
                times = torch.zeros(self.number_of_paths).unsqueeze(1)
                self.backward = False
            else:
                paths = self.final_distribution.sample([self.number_of_paths]) # START DISTRIBUTIONS
                times = torch.zeros(self.number_of_paths).unsqueeze(1) + self.T
                self.backward = True
        else:
            if forward:
                paths = self.initial_distribution.sample([self.number_of_paths]) # DATA
                times = torch.zeros(self.number_of_paths).unsqueeze(1)
                self.backward = False
            else:
                paths = self.final_distribution.sample([self.number_of_paths]) # START DISTRIBUTIONS
                times = torch.zeros(self.number_of_paths).unsqueeze(1) + self.T
                self.backward = True

        if self.backward:
            #CHECK IF THE NUMBER OF TIMES STEPS ARE T + 1
            self.time_grid_ = torch.flip(self.time_grid, [0])
        else:
            self.time_grid_ = self.time_grid

        return paths, times

    def sample_paths(self,sinkhorn_index):
        """
        Here we handle the sinkhorn iteration index
        in order to choose the correct start and process

        :param sinkhorn_index:
        :return:
        """
        if sinkhorn_index == 0:
            corresponding_process = self.reference_process
        else:
            corresponding_process = self.reference_process

        paths, times = self.sample_start_of_path(sinkhorn_index)
        paths, times = corresponding_process(paths, times)
        return paths,times

    def training_backward_rate_continuous(self,paths,times,sinkhorn_index):
        # ============================================================================
        # TRAINING
        # ============================================================================
        loss_history = []
        time0 = datetime.now()
        optimizer = Adam(self.PHI_1.parameters(), lr=self.estimate_iterations_lr)
        print("#----------Start Of Estimation Time Step {0} out of {1}-------------------".format(sinkhorn_index,self.number_of_sinkhorn_iterations))
        for score_i in tqdm.tqdm(range(self.number_of_estimate_iterations)):
            optimizer.zero_grad()

            states, time_index = self.select_random_path_position_continuous(paths)
            train_loss = self.estimator_loss_continuous_time(states, time_index, sinkhorn_index)
            train_loss.backward()
            if self.clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.PHI_1.parameters(), max_norm=self.clip_max_norm)

            loss_history.append(train_loss.item())
            optimizer.step()

            print("train loss {0}".format(train_loss.item()))
            self.writer.add_scalar("train/loss_{0}".format(sinkhorn_index), train_loss.item(), score_i)
        #==============================================================================
        timef = datetime.now()
        total_time = (timef - time0).total_seconds()
        return total_time, train_loss, loss_history

    def training_backward_rate_discrete(self, paths, times, sinkhorn_index):
        # ============================================================================
        # TRAINING
        # ============================================================================
        all_loss_history = []
        time0 = datetime.now()
        for time_index in range(self.number_of_time_steps):
            if time_index > 0:
                self.PHI_1[time_index].load_state_dict(self.PHI_1[time_index-1].state_dict())

            loss_history = []
            optimizer = Adam(self.PHI_1[time_index].parameters(), lr=self.estimate_iterations_lr)
            print("#----------Start Of Estimation Time Step {0} Sinkhorn {1}-------------------".format(time_index,
                                                                                                        sinkhorn_index))
            for score_i in tqdm.tqdm(range(self.number_of_estimate_iterations)):
                optimizer.zero_grad()
                states = self.repeat_and_new(paths,time_index)
                train_loss = self.estimator_loss_discrete_time(states, time_index, sinkhorn_index)
                train_loss.backward()

                if self.clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.PHI_1[time_index].parameters(), max_norm=self.clip_max_norm)
                loss_history.append(train_loss.item())
                optimizer.step()

                print("train loss {0}".format(train_loss.item()))
                self.writer.add_scalar("train/loss_{0}_step_{1}".format(sinkhorn_index,time_index), train_loss.item(), score_i)

            all_loss_history.append(loss_history)
        # ==============================================================================
        timef = datetime.now()
        total_time = (timef - time0).total_seconds()

        return total_time, train_loss, all_loss_history

    def inference(self):
        # ============================================================================
        # SINKHORN ITERATION
        # ============================================================================
        self.initialize_inference()
        for sinkhorn_index in range(self.number_of_sinkhorn_iterations):
            paths, times = self.sample_paths(sinkhorn_index)
            #=================================================================================
            # TRAINING BACKWARD RATE
            if self.discrete_time_networks:
                total_seconds, train_loss,loss_history = self.training_backward_rate_discrete(paths,times,sinkhorn_index)
            else:
                total_seconds, train_loss, loss_history = self.training_backward_rate_continuous(paths,times,sinkhorn_index)

            print("Time for one sinkhorn")
            print(total_seconds)
            #=================================================================================
            if self.discrete_time_networks:
                for time_index in range(self.number_of_time_steps):
                    self.PHI_0[time_index].load_state_dict(self.PHI_1[time_index].state_dict())
            else:
                self.PHI_0.load_state_dict(self.PHI_1.state_dict())
            # ================================================================================
            # SAVING
            self.save_model(paths, total_seconds, sinkhorn_index, train_loss,loss_history)
            break  # JUST ONE SINKHORN

        return total_seconds, train_loss,loss_history

    #=============================================
    # MODEL UTILS
    #=============================================
    def save_model(self,paths,total_seconds,sinkhorn_index,train_loss,loss_history):
        phi_1_file = self.best_model_path + "_sinkhorn_{0}".format(sinkhorn_index)
        torch.save({"paths": paths,
                    "PHI_0":self.PHI_0,
                    "PHI_1":self.PHI_1,
                    "input_parameters":self.input_parameters,
                    "loss":train_loss.item(),
                    "loss_history":loss_history,
                    "sinkhorn_iteration":sinkhorn_index,
                    "time_sinkhorn":total_seconds}, phi_1_file)

    def load_model(self,best_model_path=None,sinkhorn_index=0):
        if best_model_path is None:
            phi_1_file = self.best_model_path + "_sinkhorn_{0}".format(sinkhorn_index)
        else:
            phi_1_file = best_model_path + "_sinkhorn_{0}".format(sinkhorn_index)
        RESULTS = torch.load(phi_1_file)
        self.PHI_0.load_state_dict(RESULTS["PHI_0"].state_dict())
        self.PHI_1.load_state_dict(RESULTS["PHI_1"].state_dict())
        return RESULTS

if __name__=="__main__":
    p = torch.sigmoid(torch.randn((3,)))
    bs = bernoulli_spins(p)
    spins_sample = bs.sample([10])

