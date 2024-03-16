import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

from deep_fields.data.crypto.create_dataset_files import fix_dates
from deep_fields import data_path
from scipy.interpolate import interp1d
from deep_fields.data.utils import divide_data
from deep_fields.data.crypto.create_dataset_files import create_merged_dataframe, read_and_fix
from deep_fields.data.crypto.dataloaders import CryptoDataLoader


def read_full_raw(path_to_data):
    span = "full"
    ds_type = "train"
    ecosystem_train = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span, ds_type))))
    ecosystem_index_train = torch.Tensor(
        np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span, ds_type)))).long()

    ds_type = "val"
    ecosystem_val = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span, ds_type))))
    ecosystem_index_val = torch.Tensor(
        np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span, ds_type)))).long()

    ds_type = "test"
    ecosystem_test = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span, ds_type))))
    ecosystem_index_test = torch.Tensor(
        np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span, ds_type)))).long()

    ecosystem_meta = json.load(open(os.path.join(path_to_data, '{0}_meta_ecosystem.json'.format(span)), "r"))

    ecosystem_pmv = torch.cat([ecosystem_train, ecosystem_test, ecosystem_val], axis=0)
    ecosystem_index = torch.cat([ecosystem_index_train,
                                 ecosystem_index_test,
                                 ecosystem_index_val])

    return ecosystem_pmv,ecosystem_meta,ecosystem_index

def interpolation_clean(ecosystem_pmv,
                        ecosystem_meta,
                        ecosystem_index,
                        date_string,
                        initial_date=datetime(2015, 1, 1),
                        expected_not_nans = 100,
                        span="full"):
    final_date = datetime.strptime(date_string, "%Y-%m-%d")
    #======================================================
    # ORDER COINS ACCORDING HOLES PROPORTION
    survival_times_ = {}
    id_to_proportion = {}
    id_proportion = []
    name_to_birth = {}
    ALL_PROPORTIONS = []

    for i, a in enumerate(ecosystem_meta):
        survival_time = a['survival_time']
        not_nans = a['not_nans']
        proportion = not_nans / float(survival_time)
        ALL_PROPORTIONS.append(proportion)

        id_to_proportion[i] = proportion
        id_proportion.append((proportion, i))
        name_to_birth[a["id"]] = a["birth_date"]

        survival_times_[i] = survival_time

    id_proportion.sort()

    SELECTED_IDS = [a[1] for a in id_proportion]
    SELECTED_IDS_NAMES = [ecosystem_meta[a]['id'] for a in SELECTED_IDS]
    birth_dates = [datetime.strptime(ecosystem_meta[a]["birth_date"], "%Y-%m-%d %H:%M:%S") \
                   for a in SELECTED_IDS]
    last_dates = [datetime.strptime(ecosystem_meta[a]["last_date"], "%Y-%m-%d %H:%M:%S") \
                  for a in SELECTED_IDS]
    SURVIVAL_TIME = [ecosystem_meta[a]["survival_time"] for a in SELECTED_IDS]

    birth_index = [(birth_date - initial_date).days for birth_date in birth_dates]
    last_index = [(last_index_ - initial_date).days for last_index_ in last_dates]

    #======================================================

    ids_relabels = {}
    problematic_ids = []
    final_ids = []

    new_id = 0
    for j, id_ in enumerate(SELECTED_IDS):
        pmv_index = 0
        price = ecosystem_pmv[id_][birth_index[j]:last_index[j] + 1, pmv_index]
        indexes_full = np.asarray(list(range(birth_index[j], last_index[j] + 1)))
        indexes_not_nan = indexes_full[np.where(price != 0.)[0]]

        if len(indexes_not_nan) > expected_not_nans:
            price = price[np.where(price != 0.)[0]]

            f2 = interp1d(indexes_not_nan, price, kind='linear')

            min_index_not_nan = min(indexes_not_nan)
            max_index_not_nan = max(indexes_not_nan)
            indexes_full_ = indexes_full[np.where(indexes_full >= min_index_not_nan)]
            indexes_full_ = indexes_full_[np.where(indexes_full_ <= max_index_not_nan)]
            price_corrected = f2(indexes_full_)

            ecosystem_pmv[id_][indexes_full_, pmv_index] = torch.Tensor(price_corrected)

            for pmv_index in [1,2]:
                price = ecosystem_pmv[id_][:, pmv_index]
                not_zero_price = np.where(price != 0)[0]
                if len(not_zero_price) > 2:
                    birth_index_ = min(not_zero_price)
                    last_index_ = max(not_zero_price)
                    price = price[birth_index_:last_index_]

                    indexes_full = np.asarray(list(range(birth_index_, last_index_)))
                    where_not_price = np.where(price != 0.)[0]
                    indexes_not_nan = indexes_full[where_not_price]
                    price = price[np.where(price != 0.)[0]]

                    min_index_not_nan = min(indexes_not_nan)
                    max_index_not_nan = max(indexes_not_nan)
                    indexes_full_ = indexes_full[np.where(indexes_full >= min_index_not_nan)]
                    indexes_full_ = indexes_full_[np.where(indexes_full_ <= max_index_not_nan)]

                    f2 = interp1d(indexes_not_nan, price, kind='linear')
                    price_corrected = f2(indexes_full_)
                    ecosystem_pmv[id_][indexes_full_, pmv_index] = torch.Tensor(price_corrected)

            final_ids.append(id_)
            ids_relabels[id_] = new_id
            new_id+=1

        if j % 500 == 0:
            print(j)

    final_ids = torch.Tensor(final_ids).long()
    ecosystem_pmv = ecosystem_pmv[final_ids]
    ecosystem_index = ecosystem_index[final_ids].numpy().tolist()
    ecosystem_meta = list(ecosystem_meta[i.item()] for i in final_ids)
    ecosystem_index = torch.Tensor([ids_relabels[old_id] for old_id in ecosystem_index]).long()

    #==============================================================
    # DIVIDE AND REWRITE
    #==============================================================
    train_datafile = os.path.join(data_folder, "{0}_ecosystem_train_interpol".format(span))
    test_datafile = os.path.join(data_folder, "{0}_ecosystem_test_interpol".format(span))
    val_datafile = os.path.join(data_folder, "{0}_ecosystem_val_interpol".format(span))

    index_train_datafile = os.path.join(data_folder, "{0}_index_ecosystem_train_interpol".format(span))
    index_test_datafile = os.path.join(data_folder, "{0}_index_ecosystem_test_interpol".format(span))
    index_val_datafile = os.path.join(data_folder, "{0}_index_ecosystem_val_interpol".format(span))

    metadatafile = os.path.join(data_folder, "{0}_meta_ecosystem_interpol.json".format(span))

    train_pmv, test_pmv, validation_pmv = divide_data(ecosystem_pmv)
    train_index, test_index, validation_index = divide_data(ecosystem_index)

    np.save(train_datafile, train_pmv.numpy())
    np.save(test_datafile, test_pmv.numpy())
    np.save(val_datafile, validation_pmv.numpy())

    np.save(index_train_datafile, train_index.numpy())
    np.save(index_test_datafile, test_index.numpy())
    np.save(index_val_datafile, validation_index.numpy())

    #meta
    ecosystem_meta = list(map(fix_dates, ecosystem_meta))
    json.dump(ecosystem_meta, open(metadatafile, "w"))

    return final_ids,ecosystem_pmv,ecosystem_meta,ecosystem_index

if __name__=="__main__":
    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)

    ecosystem_pmv,ecosystem_meta,ecosystem_index = read_full_raw(data_folder)

    problematic_ids,ecosystem_pmv,ecosystem_meta,ecosystem_index = interpolation_clean(ecosystem_pmv, ecosystem_meta, ecosystem_index, date_string)
