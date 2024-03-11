import numpy as np
import torch
from torch.distributions import Gamma,Poisson,Uniform
from bdp.utils.debugging import timeit

#@timeit
def new_kernel(locations, location_index, new_location, number_of_arrivals, K, kernel):
    if isinstance(locations, np.ndarray):
        locations = torch.Tensor(locations)
    if isinstance(new_location,np.ndarray):
        new_location = torch.Tensor(new_location)

    index_left = list(range(number_of_arrivals))
    index_left.remove(location_index)

    k_new = K[location_index, :].copy()
    K_new = K.copy()

    new_locations = torch.cat([locations[0:location_index], locations[location_index + 1:]])

    k01 = kernel(new_location, new_locations).evaluate().detach().numpy()
    k00 = kernel(new_location, new_location).evaluate().detach().numpy()
    k_new[index_left] = k01
    k_new[location_index] = k00

    K_new[location_index, :] = k_new.copy()
    K_new[:, location_index] = k_new.T.copy()

    return K_new

if __name__=="__main__":
    print("Poisson!")


