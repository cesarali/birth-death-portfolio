import os
import torch
import numpy as np
import pandas as pd
from pprint import pprint
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Bernoulli, Normal, Exponential

from torch import matmul as m
from deep_fields.models.utils.basic_setups import create_dir_and_writer
from deep_fields.models.generative_models.diffusion_probabilistic.utils import get_timestep_embedding
from deep_fields.models.generative_models.diffusion_probabilistic.discrete.ising_utils import obtain_all_spin_states
from deep_fields.models.generative_models.diffusion_probabilistic.discrete.discrete_schrodinger import discrete_schrodinger_bridge
from deep_fields.models.generative_models.diffusion_probabilistic.discrete.ising_utils import nested_lexicographical_order, spin_state_counts, obtain_new_spin_states
from deep_fields.models.generative_models.diffusion_probabilistic.discrete.ising_utils import spin_states_stats

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class bernoulli_spins():
    """
    Spin probabilities based on poisson distribution
    """
    def __init__(self,p):
        self.p = p
        self.bernoulli_distribution = Bernoulli(p)

    def sample(self,sample_shape):
        sample_one_zeros = self.bernoulli_distribution.sample(sample_shape=sample_shape)
        sample_one_zeros[torch.where(sample_one_zeros == 0)] = -1.
        return sample_one_zeros

# initialize state
def initialize_model(number_of_spins,number_of_paths,J_mean,J_std):
    # define random interactions (fields in the diagonal)
    J = Normal(J_mean,J_std)
    J = J.sample(sample_shape=(number_of_spins,number_of_spins))
    J = (J + J.T)*.5
    return J

# hamiltonian
def Hamiltonian(J,x):
    H = m(x,J[None,:,:]).squeeze()
    H = torch.einsum('bi,bi->b',H,x)
    return H

# neighbors energy change
def Hamiltonian_i(J,x,i_random):
    J_i = J[:,i_random].T
    H_i = J[i_random,i_random] + torch.einsum('bi,bi->b',J_i,x)
    return H_i

def Hamiltonian_diagonal(J,J_d,x,i_random):
    J_i = J[:,i_random].T
    H_i = J_d[i_random] + torch.einsum('bi,bi->b',J_i,x)
    return H_i

class ising_schrodinger(discrete_schrodinger_bridge):
    """
    We use as reference process the glauber dynamics
    """
    def __init__(self,results_path=None,**kwargs):
        super().__init__(experiment_class="ising",results_path=results_path,**kwargs)

    @classmethod
    def get_parameters(cls):
        args = super().get_parameters()
        ising_arguments = {"number_of_spins": 4,
                           "beta":0.01,
                           "mu": 1.,
                           "couplings_deterministic": 1.,
                           "couplings_sigma": 1.,
                           "p0": 0.2,
                           "p1": 0.8,
                           "obtain_partition_function":True}
        args.update(ising_arguments)
        args.update({"number_of_paths":10})
        return args

    @classmethod
    def set_parameters(self,**kwargs):
        super().set_parameters(self,**kwargs)

        self.number_of_spins = kwargs.get("number_of_spins")
        self.beta = kwargs.get("beta")
        self.couplings_sigma = kwargs.get("couplings_sigma")
        self.couplings_deterministic = kwargs.get("couplings_deterministic")

        self.mu = kwargs.get("mu")
        self.p0 = torch.Tensor([kwargs.get("p0")]*self.number_of_spins)
        self.p1 = torch.Tensor([kwargs.get("p1")]*self.number_of_spins)
        self.obtain_partition_function = kwargs.get("obtain_partition_function")

        if not self.discrete_time_networks:
            self.phi_parameters['input_dim'] = self.number_of_spins + self.time_embedding_dim
        else:
            self.phi_parameters['input_dim'] = self.number_of_spins

    def define_deep_models(self,**kwargs):
        super().define_deep_models()
        assert len(self.p0) == len(self.p1) == self.number_of_spins

        #INITIAL AND FINAL DISTRIBUTIONS THIS IS THE DATA
        self.initial_distribution = bernoulli_spins(self.p0)
        self.final_distribution = bernoulli_spins(self.p1)

        #INDICES AND MASK
        self.upper_diagonal_indices = torch.triu_indices(self.number_of_spins,  self.number_of_spins, 1)
        self.lower_diagonal_indices = torch.tril_indices(self.number_of_spins, self.number_of_spins, -1)
        self.number_of_couplings =  self.upper_diagonal_indices[0].shape[0]
        self.flip_mask = torch.ones((self.number_of_spins, self.number_of_spins))
        self.flip_mask.as_strided([self.number_of_spins], [self.number_of_spins + 1]).copy_(torch.ones(self.number_of_spins) * -1.)

        # INITIALIZING COUPLINGS AND FIELDS
        if self.couplings_deterministic is None:
            if self.couplings_sigma is None:
                # we define the distributions in such a way that guarantee ergodicity of the glauber dynamics (std=1/number_of_spins)
                couplings = torch.clone(torch.Tensor(size=(
                    self.number_of_couplings,)).normal_(0., 1/float(self.number_of_spins)))
                fields = torch.clone(torch.Tensor(size=(
                    self.number_of_spins,)).normal_(0., 1/float(self.number_of_spins)))
            else:
                couplings = torch.clone(torch.Tensor(size=(
                    self.number_of_couplings,)).normal_(0., self.couplings_sigma))
                fields = torch.clone(torch.Tensor(size=(
                    self.number_of_spins,)).normal_(0., self.couplings_sigma))
        else:
            couplings = torch.ones(size=(self.number_of_couplings,))*self.couplings_deterministic
            fields = torch.ones(size=(self.number_of_spins,))*self.couplings_deterministic

        #USED IN PARAMETRIC ISING MODEL ACTUALLY
        self.fields = nn.Parameter(fields)
        self.couplings = nn.Parameter(couplings)

        #PARTITION FUNCTION FOR DIFFERENT METRICS
        if self.obtain_partition_function:
            all_states = obtain_all_spin_states(self.number_of_spins)
            with torch.no_grad():
                self.partition_function = self.hamiltonian(all_states)
                self.partition_function = torch.exp(-self.partition_function).sum()

        #UTILS FOR STATES STATISTICS
        self.spin_stats = spin_states_stats(self.number_of_spins)
        self.spin_stats.symmetric_transition_part_(self)

    def obtain_couplings_as_matrix(self,couplings=None):
        if couplings is None:
            couplings = self.couplings
        coupling_matrix = torch.zeros((self.number_of_spins, self.number_of_spins))
        coupling_matrix[self.lower_diagonal_indices[0], self.lower_diagonal_indices[1]] = couplings
        coupling_matrix = coupling_matrix + coupling_matrix.T
        return coupling_matrix

    def hamiltonian(self, states):
        """
        Calculates the Ising Hamiltonian

        Parameters
        ----------
        states:torch.Tensor
            Ising states defined by 1 or -1 spin values

        Returns
        -------
        Hamiltonian
        """
        coupling_matrix = self.obtain_couplings_as_matrix()
        H_couplings = torch.einsum('bi,bij,bj->b', states, coupling_matrix[None, :, :], states)
        H_fields = torch.einsum('bi,bi->b', self.fields[None, :], states)
        Hamiltonian = - H_couplings - H_fields
        return Hamiltonian

    def hamiltonian_diagonal(self, states, i_random):
        coupling_matrix = self.obtain_couplings_as_matrix()
        J_i = coupling_matrix[:, i_random].T
        H_i = self.fields[i_random]
        H_i =  H_i + torch.einsum('bi,bi->b', J_i, states)
        return H_i

    def log_probability(self,states):
        return -self.hamiltonian(states).sum() + states.shape[0]*torch.log(self.partition_function)

    def reference_process(self,paths,times):
        """
        Glauber dynamics

        :param paths:
        :return:
        """
        if len(paths.shape) == 2:
            paths = paths.unsqueeze(1)
        elif len(paths.shape) == 3:
            pass
        else:
            print("Wrong Path From Initial Distribution, Dynamic not possible")
            raise Exception

        number_of_paths = paths.shape[0]
        number_of_spins = self.number_of_spins
        rows_index = torch.arange(0, number_of_paths)
        for time_index in self.time_grid[1:]:
            states = paths[:, -1, :]

            i_random = torch.randint(0, number_of_spins, (number_of_paths,))

            #EVALUATES HAMILTONIAN
            H_i = self.hamiltonian_diagonal(states, i_random)
            x_i = torch.diag(states[:, i_random])
            flip_probability = (self.tau * self.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
            r = torch.rand((number_of_paths,))
            where_to_flip = r < flip_probability

            new_states = torch.clone(states)
            index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
            new_states[index_to_change] = states[index_to_change]* -1.

            paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

            # push time forward
            times_now = times[:, -1]
            times_now = torch.full_like(times_now, time_index)
            times = torch.hstack([times, times_now[:, None]])

        return paths,times

    def backward_parametric_rates(self,tau,during_training=True):
        number_of_steps = self.number_of_time_steps

        if during_training:
            PHI = self.PHI_1[number_of_steps - tau]
        else:
            PHI = self.PHI_0[number_of_steps - tau]

        all_states_ordered = self.spin_stats.all_states_in_order
        f_ss = self.spin_stats.symmetric_transition_part
        phi_ratio = torch.sigmoid(PHI(all_states_ordered)).squeeze()
        backward_transition = (phi_ratio[None, :] / phi_ratio[:, None]) ** 2
        backward_transition = backward_transition * f_ss

        return backward_transition

    def parametric_process(self,paths,times):
        """
        so we pretty much extend the poisson in the grid, but instead of sampling
        from the number of arrivals we just check if it arrived at the gap

        :return:
        """
        if len(paths.shape) == 2:
            paths = paths.unsqueeze(1)
        else:
            print("Wrong Path From Initial Distribution, Dynamic not possible")
            raise Exception

        # Obtain Transitions
        time_step = 0
        for time_index in range(self.number_of_spins):

            # embedded time
            repeated_states, new_states = self.repeat_and_new(paths, 0)
            backward_rate = self.backward_parametric_rates(new_states,repeated_states)
            backward_rate = backward_rate.reshape(IS.number_of_paths, IS.number_of_spins)
            new_states = new_states.reshape(IS.number_of_paths, IS.number_of_spins, IS.number_of_spins)
            new_states = new_states[range(0, IS.number_of_paths), torch.min(backward_rate, axis=1).indices]

            # push time forward
            times_now = times[:, -1]
            times_now = torch.full_like(times_now, time_index)

            paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)
            times = torch.hstack([times, times_now[:, None]])

        return paths, times

    def symmetric_function_from_state(self,states):
        """
        the symmetric function is calculated to all one spin states flips
        so we get f_ss.shape[0] = self.number_of_spins*states.shape[0]
        """
        states_repeated = states.repeat_interleave(self.number_of_spins, axis=0)
        i_selection = torch.tile(torch.arange(0, self.number_of_spins), (states.shape[0],))

        coupling_matrix = self.obtain_couplings_as_matrix()
        J_i = coupling_matrix[:, i_selection].T
        H_i = self.fields[i_selection]
        H_i = H_i + torch.einsum('bi,bi->b', J_i, states_repeated)
        f_ss = self.mu * (1. / 2. * torch.cosh(H_i))

        return f_ss

    def symmetric_function_for_one_spin_flip(self, old_states, new_states=None, energy=False):
        """
        FIX FOR DIAGONAL STATES, MAKE J ZERO
        """
        i_selection = torch.tile(torch.arange(0, self.number_of_spins), (self.number_of_paths,))
        H_i = self.hamiltonian_diagonal(old_states, i_selection)
        f_ss = self.mu * (1. / 2. * torch.cosh(H_i))
        if energy:
            return f_ss.reshape(-1)
        else:
            return f_ss.sum(axis=-1)

    def new_states_and_embedding(self,states,time_selected):
        new_states = obtain_new_spin_states(states,self.flip_mask)
        repeated_states = torch.repeat_interleave(states, self.number_of_spins, dim=0)
        number_of_new_states = new_states.shape[0]

        time_selected_repeated = torch.full((number_of_new_states,), time_selected)
        time_embedding = get_timestep_embedding(time_selected_repeated,time_embedding_dim=self.time_embedding_dim)

        states_n_time = torch.cat([repeated_states, time_embedding], dim=1)
        new_states_n_time = torch.cat([new_states, time_embedding], dim=1)

        return repeated_states,new_states,states_n_time, new_states_n_time

    def select_random_path_position_continuous(self,paths):
        """
        Obtains the states and counts as well as the concatenation of the time embeddings
        """
        number_of_grid_points = self.time_grid_.shape[0]
        random_time_in_grid = torch.randint(1, number_of_grid_points, (1,)).item()

        time_selected = self.time_grid_[random_time_in_grid]
        states = paths[:, random_time_in_grid, :]
        repeated_states,new_states,states_n_time, new_states_n_time = self.new_states_and_embedding(states,time_selected)

        states = (repeated_states, new_states)
        states_n_time = (states_n_time, new_states_n_time)

        return states, states_n_time

    def repeat_and_new(self,paths,time_index):
        """
        Obtains the states and counts as well as the concatenation of the time embeddings
        """
        states = paths[:, time_index, :]
        new_states = obtain_new_spin_states(states, self.flip_mask)
        repeated_states = torch.repeat_interleave(states, self.number_of_spins, dim=0)
        states = (repeated_states, new_states)
        return states

    def initialize_inference(self):
        super().initialize_inference()

    def estimator_loss_continuous_time(self,states,states_n_time,sinkhorn_index):
        repeated_states, new_states = states
        states_n_time, new_states_n_time = states_n_time

        if sinkhorn_index == 0:
            i_selection = torch.tile(torch.arange(0, self.number_of_spins), (self.number_of_paths,))
            x_i = torch.diagonal(repeated_states[:,i_selection])
            H_i = self.hamiltonian_diagonal(repeated_states, i_selection)
            Psi_x = torch.exp(-.5*x_i*H_i)
        else:
            Psi_x = torch.sigmoid(self.PHI_0(new_states_n_time))
            Psi_x = Psi_x / torch.sigmoid(self.PHI_0(states_n_time))
            Psi_x = Psi_x.squeeze()

        phi_ratio = torch.sigmoid(self.PHI_1(new_states_n_time))
        phi_ratio = phi_ratio/torch.sigmoid(self.PHI_1(states_n_time))
        phi_ratio = phi_ratio.squeeze()

        f_ss = self.symmetric_function_for_one_spin_flip(repeated_states, energy=True)
        counted_energy = f_ss * phi_ratio * Psi_x

        return counted_energy.sum()

    def estimator_loss_discrete_time(self,states,time_index,sinkhorn_index):
        repeated_states, new_states = states
        if sinkhorn_index == 0:
            i_selection = torch.tile(torch.arange(0, self.number_of_spins), (self.number_of_paths,))
            x_i = torch.diagonal(repeated_states[:,i_selection])
            H_i = self.hamiltonian_diagonal(repeated_states, i_selection)
            Psi_x = torch.exp(-.5*x_i*H_i)
        else:
            Psi_x = torch.sigmoid(self.PHI_0[self.number_of_time_steps - time_index](new_states))
            Psi_x = Psi_x / torch.sigmoid(self.PHI_0[self.number_of_time_steps - time_index](repeated_states))
            Psi_x = Psi_x.squeeze()

        phi_ratio = torch.sigmoid(self.PHI_1[time_index](new_states))
        phi_ratio = phi_ratio/torch.sigmoid(self.PHI_1[time_index](repeated_states))
        phi_ratio = phi_ratio.squeeze()

        f_ss = self.symmetric_function_for_one_spin_flip(repeated_states, energy=True)
        counted_energy = f_ss * phi_ratio * Psi_x

        return counted_energy.sum()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Ising Hamiltonian

        Parameters
        ----------
        states:torch.Tensor
            Ising states defined by 1 or -1 spin values

        Returns
        -------
        Hamiltonian
        """
        coupling_matrix = self.obtain_couplings_as_matrix()
        H_couplings = torch.einsum('bi,bij,bj->b', states, coupling_matrix[None, :, :], states)
        H_fields = torch.einsum('bi,bi->b', self.fields[None, :], states)
        Hamiltonian = - H_couplings - H_fields
        return Hamiltonian

if __name__=="__main__":
    schrodinger_parameters = ising_schrodinger.get_parameters()
    number_of_spins = schrodinger_parameters.get("number_of_spins")

    # spins models_
    schrodinger_parameters.update({"couplings_deterministic": None,
                                   "couplings_sigma": None})
    schrodinger_parameters.update({"number_of_spins": 3, "number_of_paths": 2000})

    # process characteristics
    schrodinger_parameters.update({'T': 1000., "tau": 200.})
    schrodinger_parameters.update({'p0': 0.5, "p1": 0.9})
    schrodinger_parameters.update({'mu': 0.01, "beta": 1.})
    schrodinger_parameters.update({'discrete_time_networks': True})

    # deep_model
    phi_parameters = {'dropout': 0.4,
                      'input_dim': 1,
                      'layers_dim': [20 * number_of_spins],
                      'output_dim': 1,
                      'output_transformation': None}

    # training
    schrodinger_parameters.update({"phi_parameters": phi_parameters,
                                   "time_embedding_dim": 6,
                                   "number_of_estimate_iterations": 500,
                                   "estimate_iterations_lr": 1e-4,
                                   "clip_max_norm": None})

    pprint(schrodinger_parameters)
    IS = ising_schrodinger(**schrodinger_parameters)

    #=========================================================================
    # TEST INFERENCE
    #=========================================================================
    #discrete_schrodinger\sinkhorn_1663167098
    #IS.initialize_inference()
    #paths, times = IS.sample_paths(1)
    total_seconds, train_loss,loss_history = IS.inference()
    print(total_seconds)
    #total_seconds, train_loss, loss_history = IS.training_backward_rate(paths, times, 0)
    #print("Hello World")

    #==========================================================================
    # TEST PATHS
    #==========================================================================
    #IS.initialize_inference()
    #paths, times = IS.sample_start_of_path(training=False, forward=True)
    #paths, times = IS.reference_process(paths, times)
    #states, states_n_time = IS.select_random_path_position(paths)
    #repeated_states, new_states = states
    #f_ss = IS.symmetric_function(repeated_states, energy=True)
    #print(IS.hamiltonian(states).shape)
    #print(IS.partition_function)

    #=========================================================================
    # TEST BACKWARD TRANSITIONS
    #=========================================================================
    #from deep_fields.models_.generative_models.diffusion_probabilistic.discrete.ising_utils import spin_states_stats
    #from deep_fields import models_path

    #results_path_ = os.path.join(models_path, "discrete_schrodinger", "sinkhorn_1663851552", "best_model_sinkhorn_0")

    #IS_trained = ising_schrodinger(results_path=results_path_)
    #results_ = torch.load(IS_trained.results_path)
    #paths = results_["paths"]
    #loss_history = results_["loss_history"]
    #spin_stats = spin_states_stats(IS_trained.number_of_spins)
    #number_of_steps = paths.shape[1]

    #from_states = torch.repeat_interleave(spin_stats.all_states_in_order, spin_stats.number_of_total_states, dim=0)
    #to_states = spin_stats.all_states_in_order.repeat((spin_stats.number_of_total_states, 1))
    #IS_trained.symmetric_function_for_one_spin_flip(from_states)

