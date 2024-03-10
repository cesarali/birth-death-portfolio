from deep_fields.models.sdes.utils import arrivals_departures_deep_service_file_data
from deep_fields.models.sdes.utils import create_deep_service_format

from matplotlib import pyplot as plt
from scipy.linalg import expm
import numpy as np
import operator
import random
import copy
import os

def sample_from_rate(rate):
    if rate == 0.:
        return np.infty
    return random.expovariate(rate)

def infinitesimal_propagator(number_of_states = 3, alpha=10.,cox=False):
    """
    A random walker is created per arrivals, and the departure happens whens the walker
    reaches the last state. The more states the longer the service

    alpha the bigger the longest
    """
    if not cox:
        S = np.random.random((number_of_states, number_of_states))*alpha
        print(S)
        s1 = -np.matmul(S, np.ones(number_of_states))
        s0 = copy.copy(S.diagonal()[:])
        np.fill_diagonal(S, s1)
        s0 = np.expand_dims(s0, 1)
        Gamma = np.hstack((s0, S))
        Gamma = np.vstack((np.zeros(number_of_states + 1), Gamma))
        return S, Gamma
    else:
        number_of_states = 5
        S_cox = np.zeros((5, 5))
        np.fill_diagonal(S_cox, np.asarray([-4., -2., -3., -5., -1.]))
        S_cox[0, 1] = 3.
        S_cox[1, 2] = 1.
        S_cox[2, 3] = 2.
        S_cox[3, 4] = 4.
        s0 = -np.matmul(S_cox, np.ones(number_of_states))
        s0 = np.expand_dims(s0, 1)
        Gamma = np.hstack((s0, S_cox))
        Gamma = np.vstack((np.zeros(number_of_states + 1), Gamma))
        return S_cox, Gamma

def arrivals_on_queue(time_grid,ARRIVALS,DEPARTURES):
    arrivals_on_service = np.zeros(len(time_grid))
    ARRIVALS = np.asarray(ARRIVALS)
    for i,time in enumerate(time_grid):
        for j in range(len(ARRIVALS)):
            arrival = ARRIVALS[j]
            if arrival < time:
                departure = DEPARTURES[j]
                if departure < 0:
                    arrivals_on_service[i] += 1
                elif departure > time:
                    arrivals_on_service[i] += 1
                if ARRIVALS[j] > time:
                    break
            else:
                break
    return arrivals_on_service

class Services:

    def __init__(self, parameters):
        self.lambda_0 = parameters["lambda_0"]
        self.a = parameters["a"]
        self.delta = parameters["delta"]
        self.b = parameters["b"]
        self.id = parameters["id"]
        self.T = parameters["T"]
        self.Q = list(parameters["infinitesimal_propagator"])
        self.S = parameters["S"]
        self.theta = parameters["theta"]

    def theoretical_stimates(self,t):
        lambda_star = self.a
        lambda_0 = self.lambda_0
        beta = self.delta
        alpha = self.b
        theta = self.theta

        lambda_inf = (beta*lambda_star)/(beta - alpha)
        S_T_inv = np.linalg.inv(-self.S.T)
        I = np.eye(len(self.S))
        e_ST = expm(S.T * t)

        A = I - e_ST
        A = np.matmul(A,theta)
        A = lambda_inf*np.matmul(S_T_inv,A)

        B = np.linalg.inv(S.T + (beta - alpha)*I)
        B = (lambda_0 - lambda_inf)*B

        C = np.exp(-(beta-alpha)*t)*I - e_ST
        C = np.matmul(C,theta)

        B = np.matmul(B,C)
        return np.sum(A - B)

    def simulate_hawkes(self):
        lambda_Tk_plus = self.lambda_0
        ARRIVALS = [0.]

        while ARRIVALS[-1] < self.T:
            U1 = np.random.uniform()
            U2 = np.random.uniform()
            Dk = 1. + ((self.delta * np.log(U1)) / (lambda_Tk_plus - self.a))
            S_k_1 = (-(1. / self.delta)) * (np.log(Dk))
            S_k_2 = (-(1. / self.a)) * np.log(U2)

            if Dk > 0:
                S_k = min(S_k_1, S_k_2)
            else:
                S_k = S_k_2

            ARRIVALS.append(ARRIVALS[-1] + S_k)

            lambda_Tk_minus = (lambda_Tk_plus - self.a) * np.exp(-self.delta * (ARRIVALS[-1] - ARRIVALS[-2])) + self.a
            lambda_Tk_plus = lambda_Tk_minus + self.b

        print("Number of Arrivals {0}".format(len(ARRIVALS)))
        return ARRIVALS

    def phase_time_distribution(self,arrival):
        """
        We modify:
            https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html
        """
        initial_state = 0
        number_of_states = len(self.Q)
        state_space = range(len(self.Q))  # Index the state space
        clock = arrival  # Keep track of the clock
        current_state =  np.argmax(np.random.multinomial(1, self.theta)) + 1# First state (never at zero)
        while (current_state != 0) and (clock < self.T):
            # Sample the transitions
            sojourn_times = [sample_from_rate(rate) for rate in self.Q[current_state][:current_state]]
            sojourn_times += [np.infty]  # An infinite sojourn to the same state
            sojourn_times += [sample_from_rate(rate) for rate in self.Q[current_state][current_state + 1:]]
            # Identify the next state
            next_state = min(state_space, key=lambda x: sojourn_times[x])

            sojourn = sojourn_times[next_state]
            clock += sojourn
            current_state = next_state  # Transition

        if clock < self.T:
            return clock
        else:
            return -1.

    def simulation(self):
        """
        Here we follow  Queues Driven by Hawkes Process Andrew daw, Jamol Pender (2017)

        We modify:
        https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html

        Return
        ------
        Time it takes to leave the system
        """
        ARRIVALS = self.simulate_hawkes()
        SERVICE_TIMES = [self.phase_time_distribution(a) for a in ARRIVALS]
        DEPARTURES = {i:SERVICE_TIMES[i] for i in range(len(ARRIVALS))}

        return ARRIVALS,DEPARTURES

if __name__=="__main__":
    # -----------------DEFINE QUEUES SYSTEM----------------------
    T = 1000.
    temporal_support = [(0., T)]
    cox = True
    alpha = 1.
    if not cox:
        number_of_servers = 1
        S, Q = infinitesimal_propagator(number_of_servers, alpha)
        theta = np.random.dirichlet(np.ones(number_of_servers))
        queues_parameters = {"lambda_0": 2.,
                             "a": 1.,
                             "delta": 0.001,
                             "b": .01,
                             "id": 1,
                             "T": T,
                             "infinitesimal_propagator": Q,
                             "S": S,
                             "theta": theta}
    else:
        number_of_servers = 5
        S, Q = infinitesimal_propagator(number_of_servers, alpha,cox)
        theta = np.zeros(number_of_servers)
        theta[1] = 1.
        queues_parameters = {"lambda_0": 2.,
                             "a": 1.,
                             "delta": 1.,
                             "b": (3./4.),
                             "id": 1,
                             "T": T,
                             "infinitesimal_propagator": Q,
                             "S": S,
                             "theta": theta}

    service = Services(queues_parameters)

    # ------------------THEORY-------------------------------------
    time_grid = np.linspace(0.,T,100)
    theoretical_estimates = []
    for time in time_grid:
        q_theory = service.theoretical_stimates(time)
        theoretical_estimates.append(q_theory)
    theoretical_estimates = np.asarray(theoretical_estimates)
    # ------------------MONTE CARLO LOOP--------------------------
    #NMC = 250
    #print("MC simulation")
    #list_ARRIVALS_DEPARTURES = []
    #Q_average = np.zeros(100)
    #for n in range(NMC):
    #    print(n)
    #    ARRIVALS, DEPARTURES = service.simulation()
    #    ARRIVALS_DEPARTURES = {a: DEPARTURES[i] for i, a in enumerate(ARRIVALS)}
    #    list_ARRIVALS_DEPARTURES.append(ARRIVALS_DEPARTURES)
    #    on_system = arrivals_on_queue(time_grid,ARRIVALS,DEPARTURES)
    #    Q_average += on_system
    #Q_average = Q_average/NMC

    #plt.plot(np.asarray(theoretical_estimates),"r-",label="theory")
    #plt.plot(Q_average,"b-",label="mc")
    #plt.legend(loc="best")
    #plt.show()
    #print(A)

    #---------------------------------------------------------------------
    #results_dir = "C:/Users/cesar/Desktop/Projects/PointProcesses/Results/Simulation/"
    #create_deep_service_format(list_ARRIVALS_DEPARTURES, temporal_support, results_dir)
    #---------------------------------------------------------------------
    data_dir = "C:/Users/cesar/Desktop/Projects/PointProcesses/Results/"
    RESULTS = np.load(data_dir + 'simulation_service_time_ras_beta_static_censor_penalty_0_gen_dim_200_dis_dim_200_0109_202017327474.npy')
    T = 1000.
    NMC = 10
    mc_index = 0
    SERVICES = []
    TRUE_SERVICES = []
    time_grid = np.linspace(0., T, 100)
    Q_average_ad = np.zeros(100)
    Q_average_sim = np.zeros(100)

    #Q_average = np.zeros(100)

    for n in range(NMC):
        DEPARTURES = {}
        ARRIVALS = RESULTS[n][:, -1]
        N = len(ARRIVALS)
        for k in range(1, N):
            if ARRIVALS[k] == 0.:
                break
        right_indexes = range(k)
        print(len(right_indexes))
        all_services = RESULTS[n][right_indexes, :-2]
        all_services = all_services*1000.
        all_services = all_services[:,mc_index]
        #all_services = list(all_services.mean(1))
        true_service = RESULTS[n][right_indexes, -2]*1000.
        TRUE_SERVICES.extend(list(true_service))

        right_arrivals = ARRIVALS[right_indexes]
        right_departures = RESULTS[n][right_indexes, mc_index]

        DEPARTURES = {j: right_arrivals[j] + d for j, d in enumerate(true_service)}
        arrivals_and_departures = {right_arrivals[j]: right_arrivals[j] + d for j, d in enumerate(true_service)}
        ARRIVALS = right_arrivals
        on_system = arrivals_on_queue(time_grid, ARRIVALS, DEPARTURES)
        Q_average_sim += on_system

        DEPARTURES = {j: right_arrivals[j] + d for j, d in enumerate(all_services)}
        arrivals_and_departures = {right_arrivals[j]: right_arrivals[j] + d for j, d in enumerate(all_services)}
        ARRIVALS = right_arrivals
        on_system = arrivals_on_queue(time_grid, ARRIVALS, DEPARTURES)
        Q_average_ad += on_system

    Q_average_ad = Q_average_ad / NMC
    Q_average_sim = Q_average_sim / NMC

    data_dir = "C:/Users/cesar/Desktop/Projects/PointProcesses/Results/Q/"
    np.save(os.path.join(data_dir,"Q_ad"),Q_average_ad)
    np.save(os.path.join(data_dir, "Q_sim"), Q_average_sim)
    np.save(os.path.join(data_dir, "Q_theory"), theoretical_estimates)

    plt.plot(Q_average_sim,"g*",label="simulation")
    plt.plot(theoretical_estimates,"r-",label="theory")
    plt.plot(Q_average_ad, "b-",label="adversarial")
    plt.legend(loc="best")
    plt.show()