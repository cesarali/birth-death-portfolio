import os
import sys
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from scipy.stats import invwishart
from numpy.linalg import inv, det

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def obtain_location_index_to_realization(birth_numbers):
    if isinstance(birth_numbers,torch.Tensor):
        birth_numbers = list(birth_numbers.long().numpy())
    birth_numbers = birth_numbers
    number_of_realizations = len(birth_numbers)
    location_index_to_realization = {}
    location_index = 0
    for time_index in range(number_of_realizations):
        number_of_briths_now = birth_numbers[time_index]
        for births_ in range(number_of_briths_now):
            location_index_to_realization[location_index] = time_index
            location_index += 1
    return location_index_to_realization

def obtain_blocks(time_index, assets_in_the_market, covariance_matrix, inverse_covariance_history):
    """
    Obtains matrix blocks from the growth process such as to
    speed up the calculation of the log likelihood

    Parameters
    ----------

    Return
    ------
        block_00,block_01,block_10,block_11 = np.array,np.array,np.array,np.array

        block_00.shape = (previous_number_of_assets,:previous_number_of_assets)
        block_01.shape = previous_number_of_assets,current_number_of_assets - previous_number_of_assets
                         previous_number_of_assets,birth_at_index
        block_10.shape = current_number_of_assets - previous_number_of_assets,previous_number_of_assets
                         birth_at_index,previous_number_of_assets

        block_11.shape = current_number_of_assets - previous_number_of_assets,current_number_of_assets - previous_number_of_assets
                         birth_at_index,birth_at_index
    """
    assert time_index > 0
    previous_time_index = time_index - 1

    previous_number_of_assets = assets_in_the_market[previous_time_index]
    current_number_of_assets = assets_in_the_market[time_index]

    if previous_number_of_assets < current_number_of_assets:
        previous_covariance_matrix = covariance_matrix[:previous_number_of_assets, :previous_number_of_assets]

        block_00 = previous_covariance_matrix
        block_01 = covariance_matrix[:previous_number_of_assets, previous_number_of_assets:current_number_of_assets]
        block_10 = covariance_matrix[previous_number_of_assets:current_number_of_assets,:previous_number_of_assets]

        block_11 = covariance_matrix[previous_number_of_assets:current_number_of_assets,
                                     previous_number_of_assets:current_number_of_assets]

    return block_00, block_01, block_10, block_11

def invert_and_determintant_for_blocks(block_00_determinant,block_00_inverse, block_00, block_01, block_10, block_11,test=False):
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
    old_block_size = block_00.shape[0]
    added_block_size = block_11.shape[0]
    new_block_size = old_block_size + added_block_size

    A_inv = block_00_inverse
    A = block_00
    B = block_01
    C = block_10
    D = block_11

    blocks_multiplications = D - np.dot(C, np.dot(A_inv, B))
    blocks_inverse = inv(blocks_multiplications)

    # INVERSION
    result_00 = np.dot(np.dot(blocks_inverse, C), A_inv)
    result_00 = np.dot(A_inv, np.dot(B, result_00))
    result_00 = A_inv + result_00

    result_01 = -np.dot(A_inv, np.dot(B, blocks_inverse))
    result_10 = -np.dot(np.dot(blocks_inverse, C), A_inv)

    result_11 = blocks_inverse

    final_inverse_covariance = np.zeros((new_block_size, new_block_size))

    final_inverse_covariance[:old_block_size, :old_block_size] = result_00
    final_inverse_covariance[:old_block_size, old_block_size:] = result_01
    final_inverse_covariance[old_block_size:, :old_block_size] = result_10
    final_inverse_covariance[old_block_size:, old_block_size:] = result_11

    ## DETERMINANT
    determinant = block_00_determinant * det(blocks_multiplications)

    if test:
        block_0 = np.hstack([block_00,block_01])
        block_1 = np.hstack([block_10,block_11])
        complete_block = np.vstack([block_0,block_1])
        print("Real Determinant")
        print(det(complete_block))
        print("Estimated Determinant")
        print(determinant)
        print("Inverse Multiplication Entry")
        print(np.dot(complete_block,final_inverse_covariance)[-1,-1])

    return final_inverse_covariance, determinant

def calculate_determinant_and_inverse_covariance_history(assets_in_the_market, covariance_matrix,test=False):
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
    determinants_history = np.zeros(number_of_realizations)
    inverse_covariance_history = np.zeros((number_of_realizations, total_assets_in_history, total_assets_in_history))

    # start of calculations
    current_number_of_assets = assets_in_the_market[0]
    current_covariance_matrix = covariance_matrix[:current_number_of_assets, :current_number_of_assets]
    current_determinant = det(current_covariance_matrix)
    current_inverse = inv(current_covariance_matrix)

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

            final_inverse_covariance, determinant = invert_and_determintant_for_blocks(block_00_determinant,
                                                                                       block_00_inverse,
                                                                                       block_00,
                                                                                       block_01,
                                                                                       block_10, block_11,test)

            determinants_history[time_index] = determinant
            inverse_covariance_history[time_index][:current_number_of_assets,:current_number_of_assets] = final_inverse_covariance

        # no birth of assets at this point, hence determinants and inverses remin the same
        else:
            determinants_history[time_index] = determinants_history[previous_time_index]
            inverse_covariance_history[time_index][:current_number_of_assets, :current_number_of_assets] = \
            inverse_covariance_history[previous_time_index][:current_number_of_assets, :current_number_of_assets]


    return determinants_history, inverse_covariance_history


if __name__=="__main__":
    total_assets_in_history = 30
    birth_numbers = [5, 1, 2, 2, 2, 4, 2, 2, 5, 5]
    birth_numbers = np.array(birth_numbers)
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
    covariance_matrix = IW.rvs()
    mu = torch.Tensor(np.ones(total_assets_in_history) * 0.01)

    # generate data
    distribution = MultivariateNormal(torch.Tensor(mu), torch.Tensor(covariance_matrix))
    sample = distribution.sample(sample_shape=(9,))

    determinants_history, inverse_covariance_history = calculate_determinant_and_inverse_covariance_history(assets_in_the_market, covariance_matrix,True)
