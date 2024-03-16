import os
from pprint import pprint
from deep_fields.utils.send_mails import SimpleMail
from deep_fields.models.crypto.predictors import CryptoSeq2Seq
from deep_fields.data.crypto.dataloaders import CryptoDataLoader
from deep_fields.models.crypto.predictors import CryptoSeq2NeuralSDE

from deep_fields.models.metalearning.abstract_metalearning import abstract_metalearning
from deep_fields.models.metalearning.utils_folders import strings_for_copy_from_cluster


if __name__=="__main__":
    cluster = True
    from deep_fields import models_path, data_path

    seq = False
    seqNeural = True

    # OBTAIN DATA LOADERS
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, "2021-06-02")

    kwargs = {"path_to_data":data_folder,
              "batch_size": 29,
              "steps_ahead":10,
              "span":"month"}

    data_loader = CryptoDataLoader('cpu', **kwargs)
    data_batch = next(data_loader.train.__iter__())

    if seq:
        # DEFINE BASE PARAMETERS TO BE CHANGED IN THE METASEARCH
        model_param = CryptoSeq2Seq.get_parameters()
        inference_param = CryptoSeq2Seq.get_inference_parameters()
        model_name = "crypto_seq2seq"
    elif seqNeural:
        model_param = CryptoSeq2NeuralSDE.get_parameters()
        inference_param = CryptoSeq2NeuralSDE.get_inference_parameters()
        model_name = "crypto_seq2nsde"


    """
    parameters = {"dropout": .4,
                  "dimension": 3,  # price, market cap, volume (pmv)
                  "initial_noise_size": 5,  # How many noise dimensions to sample at the start of the SDE.
                  "noise_size": 3,  # How many dimensions the Brownian motion has.
                  "hidden_size": 32,  # How big the hidden size of the generator SDE and the discriminator CDE are.
                  "mlp_size": 16,  # How big the layers in the various MLPs are.
                  "num_layers": 1,  # How many hidden layers to have in the various MLPs.
                  "steps_ahead": 4,
                  "conditional": False,
                  "conditional_hidden_state": 9,
                  "conditional_init": True,
                  "kernel_size": 3,  # encoder TCN values
                  "number_of_levels": 10,  # encoder
                  "time_encoding": 8,  # encoder
                  "model_path": os.path.join(project_path, 'results')}
    """

    #inference_param.update({"cuda": "cuda"})
    inference_param.update({"number_of_epochs": 4})
    # DEFINE SEARCH PARAMETERS
    param_search = [{"time_encoding":{"search_type": "grid",
                                      "values_list": [8,5],
                                      "check_param_dependance": False,
                                      "param_type": "model"}},
                    {"decoder_hidden_state": {"search_type": "grid",
                                              "values_list": [10,8],
                                              "check_param_dependance": False,
                                              "param_type": "model"}}]
                    #{"number_of_levels": {"search_type": "grid",
                    #                      "values_list": [10, 5],
                    #                      "check_param_dependance": False,
                    #                      "param_type": "model"}},
                    #{"learning_rate": {"search_type": "grid",
                    #                          "values_list": [1e-3,1e-6],
                    #                          "check_param_dependance": False,
                    #                          "param_type": "inference"}}]

    #'learning_rate': 0.0001,
    metalearning_parameters = {"model_path": models_path,
                               "model_name": model_name,
                               "metalearning_type": "grid_search",
                               "models_parameters_base": model_param,
                               "inference_parameters_base": inference_param,
                               "parameters_search": param_search}

    #=================================================================
    # TO STUDY RESULTS
    #=================================================================
    AM  = abstract_metalearning({"crypto_ecosystem": data_loader},**metalearning_parameters)
    AM.metalearning(True)

    #=================================================================
    # PRINT RESULTS
    #=================================================================
    rows = [("tokken", [(0, 5)]),("tokken_2", [(0, 8)])]
    columns = (1,
               [10,8],
               [])
    """
    Parameters
    ----------
    rows: list
        #[(name of row,[(metaparameters_index (in param_search),metaparameter_value)])]
    columns: tuple
        (metaparameters_index (in param_search),**
        [metaparameters_values],*
        [(metaparameters_index (in param_search)]) ***

    We are going to check values (*) for the parameter stated in (**) constrained via (***)
    """
    # [(name of row,[(metaparameters_index (in param_search),metaparameter_value)])]

    body = ""

    results_name = "loss"
    results_dataframe, results_dict_model = AM.results_table(rows, columns, results_name=results_name)
    file_name2 = "results_{0}.csv".format(results_name)
    results_dataframe.to_csv(file_name2)
    print(results_dataframe)

    body_strings = strings_for_copy_from_cluster(results_dict_model, results_name,cluster)
    body += body_strings + results_dataframe.to_latex() + "\n\n"

    pprint(body_strings)
    #========================================================
    # SEND MAIL
    #========================================================
    #subject = "Crypto Sequence2Sequence"
    #receiver_email = "ojedamarin@tu-berlin.de"
    #filenames = [file_name2]
    #mail_object = SimpleMail()
    #mail_object.send_mail(receiver_email, subject, body, filenames)