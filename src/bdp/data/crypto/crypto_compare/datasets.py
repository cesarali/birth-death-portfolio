import os
import sys
import json
import pickle
from collections import namedtuple, defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
#pmv price marketcap and volumes
pmv_nametupled_full = namedtuple('data', 'ids, pmv, birth, lifetime,max')
pmv_nametupled = namedtuple('data', 'ids, pmv, max')

class PortfolioDataset(Dataset):
    """
    This class aggregates all data (TO SELECT FROM ALL AVAILABLE COINS)
    in order to have one iterator over all prices for
    """
    data: dict
    def __init__(self, path_to_data: str, date_string: str,span="full",clean=None):
        super(PortfolioDataset, self).__init__()
        assert os.path.exists(path_to_data)
        self.span = span
        if clean is not None:
            ds_type = "train"
            self.ecosystem_train = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean))))
            self.ecosystem_index_train = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean)))).long()

            ds_type = "test"
            self.ecosystem_test = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean))))
            self.ecosystem_index_test = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean)))).long()

            ds_type = "val"
            self.ecosystem_val = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean))))
            self.ecosystem_index_val = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean)))).long()
        else:
            ds_type = "train"
            self.ecosystem_train = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span,ds_type))))
            self.ecosystem_index_train = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span,ds_type)))).long()

            ds_type = "test"
            self.ecosystem_test = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span,ds_type))))
            self.ecosystem_index_test = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span,ds_type)))).long()

            ds_type = "val"
            self.ecosystem_val = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span,ds_type))))
            self.ecosystem_index_val = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span,ds_type)))).long()


        self.ecosystem = torch.cat([self.ecosystem_train,
                                    self.ecosystem_test,
                                    self.ecosystem_val],axis=0)

        self.ecosystem_index = torch.cat(
            [self.ecosystem_index_train,
             self.ecosystem_index_test,
             self.ecosystem_index_val]
        )

        if self.span == "full":
            initial_date = datetime(2015, 1, 1)
            final_date = datetime.strptime(date_string,"%Y-%m-%d")

            self.ecosystem_meta = json.load(
                open(os.path.join(path_to_data, '{0}_meta_ecosystem.json'.format(span)), "r"))
            self.ecosystem_lifetimes = [self.ecosystem_meta[coin_index]["survival_time"] for coin_index in
                                        self.ecosystem_index]
            self.ecosystem_lifetimes = torch.Tensor(self.ecosystem_lifetimes).long()

            birth_dates = [datetime.strptime(self.ecosystem_meta[coin_index]["birth_date"],"%Y-%m-%d %H:%M:%S")
                           for coin_index in self.ecosystem_index]
            self.birth_index = torch.Tensor([(birth_date - initial_date).days for birth_date in birth_dates]).long()

            self.initial_prices = torch.index_select(self.ecosystem[:,:,0], 1, self.birth_index)
            self.initial_prices = torch.diagonal(self.initial_prices)

        self.max_ = torch.max(self.ecosystem, dim=1)
        self.max_ = self.max_.values[:, None, :]
        self.ecosystem = self.ecosystem / (self.max_ + 1e-6) # wrong
        assert (self.ecosystem  != self.ecosystem ).sum() < 1 #NO NANS

    def __getitem__(self, i):
        if self.span == "full":
            return pmv_nametupled_full(self.ecosystem_index[i],self.ecosystem[i],self.birth_index[i],
                                       self.ecosystem_lifetimes[i],self.max_[i])
        else:
            return pmv_nametupled(self.ecosystem_index[i],self.ecosystem[i],self.max_[i])

    def __len__(self):
        return self.ecosystem.shape[0]

class EcosystemDataset(Dataset):
    """
    We normalize the data set by its biggest value
    """
    data: dict

    def __init__(self, path_to_data: str, date_string: str, ds_type="train",span="full",clean=None):
        super(EcosystemDataset, self).__init__()
        assert os.path.exists(path_to_data)
        self.span = span
        if clean is not None:
            self.ecosystem = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean))))
            self.ecosystem_index = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}_{2}.npy'.format(span,ds_type,clean)))).long()
        else:
            self.ecosystem = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_ecosystem_{1}.npy'.format(span,ds_type))))
            self.ecosystem_index = torch.Tensor(np.load(os.path.join(path_to_data, '{0}_index_ecosystem_{1}.npy'.format(span,ds_type)))).long()

        if self.span == "full":
            initial_date = datetime(2015, 1, 1)
            final_date = datetime.strptime(date_string,"%Y-%m-%d")

            self.ecosystem_meta = json.load(
                open(os.path.join(path_to_data, '{0}_meta_ecosystem.json'.format(span)), "r"))
            self.ecosystem_lifetimes = [self.ecosystem_meta[coin_index]["survival_time"] for coin_index in
                                        self.ecosystem_index]
            self.ecosystem_lifetimes = torch.Tensor(self.ecosystem_lifetimes).long()

            birth_dates = [datetime.strptime(self.ecosystem_meta[coin_index]["birth_date"],"%Y-%m-%d %H:%M:%S")
                           for coin_index in self.ecosystem_index]
            self.birth_index = torch.Tensor([(birth_date - initial_date).days for birth_date in birth_dates]).long()

            self.initial_prices = torch.index_select(self.ecosystem[:,:,0], 1, self.birth_index)
            self.initial_prices = torch.diagonal(self.initial_prices)

            # define birth days
            #negative_survival_index = list(map(lambda a: a * (-1), self.portfolio_survival.numpy()))
            #self.birth_index = self.portfolio_pmv[:, :, 0].shape[1] + torch.Tensor(negative_survival_index).long()
            #self.initial_prices = torch.diagonal(torch.index_select(self.portfolio_pmv[:, :, 0], 1, self.birth_index))
            #self.smallest_survival_time = int(min(self.portfolio_survival.numpy().tolist()))

        self.max_ = torch.max(self.ecosystem, dim=1)
        self.max_ = self.max_.values[:, None, :]
        self.ecosystem = self.ecosystem / (self.max_ + 1e-6) # wrong
        assert (self.ecosystem  != self.ecosystem ).sum() < 1 #NO NANS

    def __getitem__(self, i):
        if self.span == "full":
            return pmv_nametupled_full(self.ecosystem_index[i],self.ecosystem[i],self.birth_index[i],
                                       self.ecosystem_lifetimes[i],self.max_[i])
        else:
            return pmv_nametupled(self.ecosystem_index[i],self.ecosystem[i],self.max_[i])

    def __len__(self):
        return self.ecosystem.shape[0]

if __name__=="__main__":
    from deep_fields import data_path

    #'birth_'
    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)
    eco_dataset = EcosystemDataset(data_folder,date_string,"train","full","interpol")

    #print(eco_dataset[0].lifetime)
    #print(eco_dataset[0].ids)
    #print(eco_dataset[0].pmv.shape)

    portfolio_dataset = PortfolioDataset(data_folder,date_string,"full","interpol")

    #print(eco_dataset[0].ids)
    print(portfolio_dataset[0].pmv.shape)



