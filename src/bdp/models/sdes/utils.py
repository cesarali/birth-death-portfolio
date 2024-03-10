import os
import json
import numpy as np

def arrivals_departures_deep_service_file_data(ARRIVALS_DEPARTURES,time_support):
    """
    :parameter:
        ARRIVALS_DEPARTURES: dictionary obtained from queues class simulation
                             {arrival:departure}
    :return:
    """
    number_of_arrivals = len(ARRIVALS_DEPARTURES)
    ARRIVALS_DEPARTURES_ARRAY = np.zeros((number_of_arrivals,3))
    j = 0
    for arrival, departure in ARRIVALS_DEPARTURES.items():
        ARRIVALS_DEPARTURES_ARRAY[j, 0] = j
        if arrival < time_support[0][1]:
            if departure < time_support[0][1]:
                ARRIVALS_DEPARTURES_ARRAY[j,1] = arrival
                ARRIVALS_DEPARTURES_ARRAY[j,2] = departure
            else:
                ARRIVALS_DEPARTURES_ARRAY[j, 1] = arrival
                ARRIVALS_DEPARTURES_ARRAY[j, 2] = arrival - time_support[0][1]
        j+=1

    arrivals_index_sort = np.argsort(ARRIVALS_DEPARTURES_ARRAY[:,1])
    departures_index_sort = np.argsort(ARRIVALS_DEPARTURES_ARRAY[:,2])
    ARRIVALS_DEPARTURES_ARRAY_D = ARRIVALS_DEPARTURES_ARRAY[departures_index_sort,:]
    ARRIVALS_DEPARTURES_ARRAY_D = ARRIVALS_DEPARTURES_ARRAY_D[:,np.asarray([0, 2])]
    return ARRIVALS_DEPARTURES_ARRAY[arrivals_index_sort,:],ARRIVALS_DEPARTURES_ARRAY_D

def create_deep_service_format(list_of_ARRIVALS_AND_DEPARTURES,temporal_support,results_dir):
    """
    create the files format for the deep service distribution
    :return:
    """
    for batch_index,ARRIVALS_AND_DEPARTURES in enumerate(list_of_ARRIVALS_AND_DEPARTURES):
        current_folder = results_dir + str(batch_index) + "/"
        os.makedirs(current_folder)
        arrivals_cov, departures_cov = arrivals_departures_deep_service_file_data(ARRIVALS_AND_DEPARTURES,temporal_support)
        np.savetxt(current_folder+"arrivals_covariates_1_.txt",arrivals_cov)
        np.savetxt(current_folder + "departures_covariates_1_.txt",departures_cov)
        file_metadata = {"files":{1:{"num_arrivals":arrivals_cov.shape[0],"num_departures":departures_cov.shape[0]}},
                         "temporal_support":temporal_support,
                         "arrivals_file_header":{0:"arrival_id",1:"arrivals_time",2:"departure_time"},
                         "departures_file_header":{0:"departure_id",1:"departure_time"},
                         "total_arrivals":arrivals_cov.shape[0],
                         "total_departures":departures_cov.shape[0]}
        json.dump(file_metadata,open(current_folder+"metadata.json","w"))