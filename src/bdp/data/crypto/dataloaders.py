import torch
from deep_fields.data.crypto.datasets import EcosystemDataset
from datetime import datetime

import os
import sys
import json
import torch
import pandas as pd

import numpy as np
from abc import ABC

import pymongo
from collections import namedtuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from deep_fields.models.crypto.birth_stats import top_and_birth
from deep_fields.data.crypto.create_dataset_files import create_merged_dataframe
from deep_fields.data.crypto.datasets import EcosystemDataset, PortfolioDataset, pmv_nametupled, pmv_nametupled_full
from deep_fields.data.crypto.data_basics import top_from_collection

sampler = torch.utils.data.RandomSampler
DistributedSampler = torch.utils.data.distributed.DistributedSampler


class ADataLoader(ABC):
    _train_iter: DataLoader
    _valid_iter: DataLoader
    _test_iter: DataLoader

    def __init__(self, device, rank: int = 0, world_size: int = -1, **kwargs):
        self.dataset_kwargs = kwargs
        self.device = device
        self.batch_size = kwargs.get('batch_size')
        self.world_size = world_size
        self.rank = rank

    @property
    def train(self):
        return self._train_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def test(self):
        return self._test_iter

    #@property
    #def predict(self):
    #    return self._predict_iter

    @property
    def n_train_batches(self):
        return len(self.train.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_test_batches(self):
        return len(self.test.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_validate_batches(self):
        return len(self.validate.dataset) // self.batch_size // abs(self.world_size)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        return len(self.test.dataset)

    @property
    def number_of_users(self):
        try:
            return self.train_set_size + self.validation_set_size + self.prediction_set_size + self.test_set_size
        except:
            return self.train_set_size + self.validation_set_size + self.test_set_size

class CryptoDataLoader(ADataLoader):
    """
    first we create and store the examples and text fields
    here, then, the dataset is divided in prediction and training
    prediction is only in the future (respects causality)

    the training is then divided in train and validation, which is a random split
    """
    global_nametupled = namedtuple('Counts', 'atoms_alpha, atoms_beta, atoms, '
                                             'locations, weights, w, sticks, gp_parameters')

    def __init__(self, device, rank: int = 0, world_size=-1, span="full",**kwargs):
        super().__init__(device, rank, world_size, **kwargs)
        data_dir = kwargs.get("path_to_data")
        date_string = kwargs.get("date_string")
        clean  = kwargs.get("clean")

        self.span = span
        self.steps_ahead = kwargs.get("steps_ahead")

        train_dataset = EcosystemDataset(data_dir, date_string,"train",span,clean)
        test_dataset = EcosystemDataset(data_dir, date_string,"test",span,clean)
        valid_dataset = EcosystemDataset(data_dir, date_string,"val",span,clean)

        if clean is not None:
            self.meta_ecosystem = json.load(open(os.path.join(data_dir, '{0}_meta_ecosystem_{1}.json'.format(span,clean)), "r"))
        else:
            self.meta_ecosystem = json.load(open(os.path.join(data_dir, '{0}_meta_ecosystem.json'.format(span)), "r"))

        self.index_ = pd.read_csv(os.path.join(data_dir, "{0}_ecosystem_datetime.csv".format(span)), header=None)

        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, batch_size=self.batch_size)

        self.dynamic_covariates = 3
        self.covariates_dimension = 0
        self.dimension = 3

    def set_portfolio_assets(self,date="2021-06-02",span="month",predictor=None,top=10,date0=None,datef=None,max_size=10):
        """
        This is actually super important and not trivial, basically a project of its own

        :param predictor:
        :param top:
        :return:
        """
        from deep_fields import data_path
        crypto_folder = os.path.join(data_path, "raw", "crypto")
        data_folder = os.path.join(crypto_folder, date)

        initial_date = datetime(2015, 1, 1)
        final_date = datetime.strptime(date, "%Y-%m-%d")

        if predictor is None:
            if top is not None:
                if date0 is None:
                    coins_name = top_from_collection(date,max_size)
                else:
                    # Here we select the coins which are birth in a given window of time
                    top_coins_name, coins_name = top_and_birth(date, date0, datef, top=10, max_size=max_size)

                client = pymongo.MongoClient()
                db = client["crypto"]
                collection = db['birth_{0}'.format(date)]

                data_merged, coins_data = create_merged_dataframe(data_folder,
                                                                  collection,
                                                                  break_point=max_size,
                                                                  all_coins_ids=coins_name,
                                                                  span=span)

                #define prices and returns-----------------------------------------------------------
                price = torch.Tensor(data_merged["price"].values).T
                market_cap = torch.Tensor(data_merged["market_cap"].values).T
                volume = torch.Tensor(data_merged["volume"].values).T

                #define birth indexes and death indexes----------------------------------------------
                survival_time = [coin_data["survival_time"] for coin_data in coins_data]

                birth_dates = [coin_data["birth_date"] for coin_data in coins_data]
                birth_index = torch.Tensor([(birth_date - initial_date).days for birth_date in birth_dates]).long()

                not_nans = price == price
                death_index = torch.Tensor([torch.where(not_nan)[0].max() for not_nan in not_nans]).long()

                self.portfolio_final_prices = torch.index_select(price, 1, death_index)
                self.portfolio_final_prices = torch.diagonal(self.portfolio_final_prices)

                self.portfolio_initial_prices = torch.index_select(price, 1, birth_index)
                self.portfolio_initial_prices = torch.diagonal(self.portfolio_initial_prices)

                assert (self.portfolio_initial_prices != self.portfolio_initial_prices).sum().item() == 0
                assert (self.portfolio_final_prices != self.portfolio_final_prices).sum().item() == 0

                self.portfolio_ids = [coin_data["id"] for coin_data in coins_data]
                self.portfolio_survival = torch.Tensor(survival_time)
                self.smallest_survival_time = int(self.portfolio_survival.min().item())
                self.portfolio_birth = birth_dates

                self.portfolio_birth_indexes = birth_index
                self.portfolio_death_index = death_index

                #set pmv returns and corresponding masks---------------------------------------------
                self.portfolio_pmv_mask = (price == price) * (market_cap == market_cap) * (volume == volume)
                self.portfolio_returns = (price[:, 1:] - price[:, :-1]) / (price[:, :-1])
                self.portfolio_returns_mask = (self.portfolio_returns == self.portfolio_returns)
                self.portfolio_pmv = torch.cat([price.unsqueeze(-1), market_cap.unsqueeze(-1), volume.unsqueeze(-1)], dim=-1)
                self.portfolio_size = self.portfolio_pmv.shape[0]

                #remove nans and normalize-----------------------------------------------------------
                portfolio_pmv_ = self.portfolio_pmv
                portfolio_pmv_[portfolio_pmv_!= portfolio_pmv_] = 0.
                max_ = torch.max(portfolio_pmv_, dim=1)
                max_ = max_.values[:, None, :]
                self.normalized_portfolio_pmv = portfolio_pmv_ / (max_)  # include nans when the price is 0

    #def portolio_batch(self):
    #    pmv_nametupled
    #    pmv_nametupled_full(self.ecosystem_index[i], self.ecosystem[i], self.ecosystem_lifetimes[i], self.max_[i])
    #    pmv_nametupled_full

    def unfold(self,series):
        batch_size = series.shape[0]
        sequence_lenght = series.shape[1]
        dimension = series.shape[2]
        sequence_lenght_ = sequence_lenght - self.steps_ahead + 1

        # we unfold the series so at each step on time we predict (steps_ahead)
        unfolded_series = series.unfold(dimension=1, size=self.steps_ahead, step=1).contiguous()
        unfolded_series = unfolded_series.reshape(batch_size * sequence_lenght_,
                                                  dimension,
                                                  self.steps_ahead)
        #unfolded_series = unfolded_series.permute(0, 2, 1)
        return unfolded_series

    def random_portfolio_selection(self):
        test_proportions = 0.8
        train_proportions = 0.1
        validation_proportions = 0.1

        train_in_portfolio = int(train_proportions * portfolio_size)
        test_in_portfolio = int(test_proportions * portfolio_size)
        validation_in_portfolio = int(validation_proportions * portfolio_size)

class PortfolioDataLoader(ADataLoader):
    """
    first we create and store the examples and text fields
    here, then, the dataset is divided in prediction and training
    prediction is only in the future (respects causality)

    the training is then divided in train and validation, which is a random split
    """

    def __init__(self, device, rank: int = 0, world_size=-1, span="full",**kwargs):
        super().__init__(device, rank, world_size, **kwargs)
        data_dir = kwargs.get("path_to_data")
        date_string = kwargs.get("date_string")
        clean  = kwargs.get("clean")

        self.span = span
        self.steps_ahead = kwargs.get("steps_ahead")
        dataset = PortfolioDataset(data_dir, date_string,span,clean)

        if clean is not None:
            self.meta_ecosystem = json.load(open(os.path.join(data_dir, '{0}_meta_ecosystem_{1}.json'.format(span,clean)), "r"))
        else:
            self.meta_ecosystem = json.load(open(os.path.join(data_dir, '{0}_meta_ecosystem.json'.format(span)), "r"))

        self.index_ = pd.read_csv(os.path.join(data_dir, "{0}_ecosystem_datetime.csv".format(span)), header=None)

        train_sampler = None

        if self.world_size != -1:
            train_sampler = DistributedSampler(dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, batch_size=self.batch_size)
        self._valid_iter = None
        self._test_iter = None

        self.dynamic_covariates = 3
        self.covariates_dimension = 0
        self.dimension = 3

if __name__=="__main__":
    from deep_fields import data_path

    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, "2021-06-14")
    date_string = "2021-06-14"

    kwargs = {"path_to_data":data_folder,
              "batch_size": 29,
              "steps_ahead":10,
              "date_string": date_string,
              "clean":"interpol",
              "span":"full"}

    date0 = datetime(2018, 1, 1)
    datef = datetime(2019, 1, 1)

    portfolio_data_loader = PortfolioDataLoader('cpu', **kwargs)
    print(portfolio_data_loader.train_set_size)

    crypto_data_loader = CryptoDataLoader('cpu', **kwargs)
    print(crypto_data_loader.train_set_size)

    """
    data_loader.set_portfolio_assets("2021-06-14",
                                     "full",
                                     predictor=None,
                                     top=10,
                                     date0=date0,
                                     datef=datef,
                                     max_size=20)

    data_batch = next(data_loader.train.__iter__())

    #print(data_batch.pmv.shape)
    #print(data_batch.lifetime)
    print(data_loader.portfolio_pmv.shape)
    print(data_loader.portfolio_survival)
    """