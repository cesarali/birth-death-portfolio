import os
import sys
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from scipy.stats import invwishart
from torch import matmul as m
from torch import einsum

from torch.distributions import MultivariateNormal
from bdp.models.random_fields.blocks_utils import obtain_blocks, obtain_location_index_to_realization

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
PI = np.pi

def invert_and_determintant_for_blocks_torch(block_00_determinant,block_00_inverse, block_00, block_01, block_10, block_11,test=False):
    """
    Calculates a matrix inverse, whene the matrix is decomposed in blocks such:

    Covariance = [block_00,block_01,
                  block_10,block_11]

    Parameters
    ----------
        block_00_inverse,
        block_00,
        block_01,
        block_10,
        block_11,
        total_assets_in_history

    Returns
    -------
    covariance_inverse, covariance_determinant
    """
    assert block_00_determinant.dtype == torch.float64,"Not right type"
    assert block_00_inverse.dtype == torch.float64,"Not right type"

    assert block_00.dtype == torch.float64,"Not right type"
    assert block_01.dtype == torch.float64,"Not right type"
    assert block_10.dtype == torch.float64,"Not right type"
    assert block_11.dtype == torch.float64,"Not right type"

    old_block_size = block_00.shape[0]
    added_block_size = block_11.shape[0]
    new_block_size = old_block_size + added_block_size

    A_inv = block_00_inverse
    A = block_00
    B = block_01
    C = block_10
    D = block_11

    blocks_multiplications = D - m(C, m(A_inv, B))
    blocks_inverse = torch.inverse(blocks_multiplications).type(torch.float64)

    # INVERSION
    result_00 = m(m(blocks_inverse, C), A_inv)
    result_00 = m(A_inv, m(B, result_00))
    result_00 = A_inv + result_00

    result_01 = -m(A_inv, m(B, blocks_inverse))
    result_10 = -m(m(blocks_inverse, C), A_inv)

    result_11 = blocks_inverse

    final_inverse_covariance = torch.zeros((new_block_size, new_block_size)).type(torch.float64)

    final_inverse_covariance[:old_block_size, :old_block_size] = result_00
    final_inverse_covariance[:old_block_size, old_block_size:] = result_01
    final_inverse_covariance[old_block_size:, :old_block_size] = result_10
    final_inverse_covariance[old_block_size:, old_block_size:] = result_11

    ## DETERMINANT
    determinant = block_00_determinant * torch.det(blocks_multiplications).type(torch.float64)

    if test:
        block_0 = torch.hstack([block_00,block_01])
        block_1 = torch.hstack([block_10,block_11])
        complete_block = torch.vstack([block_0,block_1])
        print("Real Determinant")
        print(torch.det(complete_block))
        print("Estimated Determinant")
        print(determinant)
        print("Inverse Multiplication Entry")
        print(m(complete_block,final_inverse_covariance)[-1,-1])

    return final_inverse_covariance, determinant

def calculate_determinant_and_inverse_covariance_history_torch(assets_in_the_market, covariance_matrix,test=False):
    """
    here we make use of the block structure of the covariance matrix as induced by the growth processes
    in order to calculate the determinants and inverse covariances as required by the log probability
    calculation

    :return:
        determinants_history, np.array[number_of_realizations]
        inverse_covariance_history: np.array[number_of_realizations,total_assets_inhistory,total_assets_in_history],
    """
    number_of_realizations = len(assets_in_the_market)
    total_assets_in_history = assets_in_the_market[-1]

    # inverse by history (blockswise)
    determinants_history = torch.zeros(number_of_realizations).type(torch.float64)
    inverse_covariance_history = torch.zeros((number_of_realizations, total_assets_in_history, total_assets_in_history)).type(torch.float64)

    # start of calculations
    current_number_of_assets = assets_in_the_market[0]
    current_covariance_matrix = covariance_matrix[:current_number_of_assets, :current_number_of_assets]
    current_determinant = torch.det(current_covariance_matrix).type(torch.float64)
    current_inverse = torch.inverse(current_covariance_matrix).type(torch.float64)

    determinants_history[0] = current_determinant
    inverse_covariance_history[0][:current_number_of_assets, :current_number_of_assets] = current_inverse
    for time_index in range(1, number_of_realizations):
        # time index handling
        assert time_index > 0
        previous_time_index = time_index - 1
        previous_number_of_assets = assets_in_the_market[previous_time_index]
        current_number_of_assets = assets_in_the_market[time_index]

        # birth of assets
        if previous_number_of_assets < current_number_of_assets:
            block_00_determinant = determinants_history[previous_time_index]
            block_00_inverse = inverse_covariance_history[previous_time_index][:previous_number_of_assets,:previous_number_of_assets]

            block_00, block_01, block_10, block_11 = obtain_blocks(time_index,
                                                                   assets_in_the_market,
                                                                   covariance_matrix,
                                                                   inverse_covariance_history)

            final_inverse_covariance, determinant = invert_and_determintant_for_blocks_torch(block_00_determinant,
                                                                                             block_00_inverse,
                                                                                             block_00,
                                                                                             block_01,
                                                                                             block_10,
                                                                                             block_11,
                                                                                             test)

            determinants_history[time_index] = determinant
            inverse_covariance_history[time_index][:current_number_of_assets,:current_number_of_assets] = final_inverse_covariance

        # no birth of assets at this point, hence determinants and inverses remin the same
        else:
            determinants_history[time_index] = determinants_history[previous_time_index]
            inverse_covariance_history[time_index][:current_number_of_assets, :current_number_of_assets] = \
            inverse_covariance_history[previous_time_index][:current_number_of_assets, :current_number_of_assets]

    return determinants_history, inverse_covariance_history

def log_probability_no_blocks(sample,mu,number_of_realizations,assets_in_the_market,covariance_matrix):
    # OLD CALCULATION (EXACT; LOG PROB CALCULATED AT EACH STEP)
    exact_determinants = []
    exact_probability = []
    exact_bilinear = []
    for time_index in range(number_of_realizations):
        current_assets_in_the_market = assets_in_the_market[time_index]
        current_interest_rate = mu[None, :current_assets_in_the_market]
        current_diffusion_covariance = covariance_matrix[:current_assets_in_the_market, :current_assets_in_the_market]

        exact_determinants.append(torch.det(current_diffusion_covariance).type(torch.float64))

        current_inverse_covariance = torch.inverse(current_diffusion_covariance)
        vector = sample[time_index][:current_assets_in_the_market] - current_interest_rate

        exact_inverse_covariance = einsum("bi,bij,bj->b", vector, current_inverse_covariance[None, :, :], vector).item()
        exact_bilinear.append(exact_inverse_covariance)

        current_diffusion_distribution = MultivariateNormal(current_interest_rate,
                                                            current_diffusion_covariance)

        current_log_returns = sample[time_index, :current_assets_in_the_market]
        cdd = current_diffusion_distribution.log_prob(current_log_returns)

        exact_probability.append(cdd.item())
    return exact_probability

def log_probability_from_blocks(sample,mu,assets_in_the_market,determinants_history,inverse_covariance_history):
    """
    here we calculate the log probability for the whole markets realizations
    taking into account the blockwise structre of the covariance matrix as induced by
    the birth process

    :param sample:
    :param mu:
    :param assets_in_the_market:
    :param determinants_history:
    :param inverse_covariance_history:
    :return:
    """
    # LOG PROBABILITY FROM THE DETERMINANT
    log_probability_determinant = torch.log(
        (((2 * np.pi) ** assets_in_the_market.float().type(torch.float64))) * determinants_history)

    vector = sample - mu[None, :]
    log_probability_inverse_covariance = einsum("bi,bij,bj->b", vector, inverse_covariance_history, vector)
    log_probability = -.5 * (log_probability_determinant + log_probability_inverse_covariance)

    return log_probability

if __name__=="__main__":
    # ============================================================
    # TEST
    # ============================================================
    total_assets_in_history = 30
    birth_numbers = [5, 1, 2, 2, 2, 4, 2, 2, 5, 5]
    birth_numbers = np.asarray(birth_numbers)
    assets_in_the_market = birth_numbers.cumsum()
    number_of_realizations = len(assets_in_the_market)
    location_index_to_realization = obtain_location_index_to_realization(birth_numbers)

    # generate covariance matrix prior
    nu = total_assets_in_history + 1.
    Psi = np.random.rand(total_assets_in_history, total_assets_in_history)
    Psi = np.dot(Psi, Psi.transpose())

    a_J = np.ones(total_assets_in_history) * 10.
    b_J = 1.

    lambda_ = 1 / b_J
    mu_0 = a_J

    IW = invwishart(nu, Psi)
    covariance_matrix = torch.Tensor(IW.rvs())
    mu = torch.Tensor(np.ones(total_assets_in_history) * 0.01)

    covariance_matrix = covariance_matrix.type(torch.float64)
    mu = mu.type(torch.float64)

    # generate data
    distribution = MultivariateNormal(mu, covariance_matrix)
    sample = distribution.sample(sample_shape=(number_of_realizations,))
    if isinstance(assets_in_the_market, torch.Tensor):
        assets_in_the_market = list(assets_in_the_market.long().numpy())
    for i, current_number_of_assets in enumerate(assets_in_the_market):
        sample[i, current_number_of_assets:] = torch.zeros_like(sample[i, current_number_of_assets:])
    log_returns = sample

    determinants_history, inverse_covariance_history = calculate_determinant_and_inverse_covariance_history_torch(
        assets_in_the_market, covariance_matrix)

    log_probility = log_probability_from_blocks(sample, mu, assets_in_the_market, determinants_history, inverse_covariance_history)
    print(log_probility)
