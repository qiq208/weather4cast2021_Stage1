

import argparse

import glob
import h5py
import numpy as np
from fastai.basics import *
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.hook import summary
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.fp16 import *
from fastai.data.load import DataLoader as FastDataLoader
from fastai.callback.progress import CSVLogger
from torch.utils.data import *
from collections.abc import Iterable

import utils.config as config
from utils.w4c_dataloader_aio import create_dataset
from utils.vunet_model import Net1, VUNetLoss2, valid_leaderboard, valid_leaderboard2

def get_datasets(region_ids, train_on_validation=False):
    if len(region_ids)==1:
        params = config.get_params(region_id=region_ids[0])['data_params']
        params['return_meta']=False
        valid = create_dataset('training', params)
        if train_on_validation:
            train = [create_dataset('training', params), create_dataset('validation', params)]
            train = ConcatDataset(train)
            train.n_inp = 1
        else:
            train = create_dataset('training', params)
    else:
        train=[]
        valid=[]
        for reg in region_ids:
            params = config.get_params(region_id=reg)['data_params']
            train.append(create_dataset('training', params))
            if train_on_validation:
                train.append(create_dataset('validation', params))
            valid.append(create_dataset('validation', params))
        train = ConcatDataset(train)
        train.n_inp = 1
        valid = ConcatDataset(valid)
        valid.n_inp = 1
    return train, valid

    
def train(regions, folder_to_save_models, batch_size, num_workers, device):
    """ main training method
    """
    # ------------
    # Setup Model
    # ------------
    print('Setup Model')
    Model = Net1(in_channels=8*4+3,out_channels=4*32)
    
    # ------------
    # Set up the Dataset and DataLoaders
    # ------------
    train, valid = get_datasets(region_ids, train_on_validation=False)
    train_dl = FastDataLoader(dataset=train,
                          bs=batch_size,
                          num_workers=num_workers,
                          pin_memory=True,
                          shuffle=True,
                          device=torch.device(device))
    valid_dl = FastDataLoader(dataset=valid,
                              bs=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True,
                              device=torch.device(device))
    data = DataLoaders(train_dl, valid_dl, device=torch.device(device))
    model = load_model(Model, params, checkpoint_path)
    
    # ------------
    # Trainer
    # ------------
    learn = Learner(
        data, 
        Model.to(device), 
        loss_func=VUNetLoss2, metrics=[valid_leaderboard, valid_leaderboard2], 
        model_dir=folder_to_save_models, cbs=CSVLogger)
    
    print('Start Training')
    # Train
    learn.fit_one_cycle(2, lr_max=2e-04)
    learn.save('Comb_2')
    learn.fit_flat_cos(2, lr=2e-04, pct_start=0)
    learn.save('Comb_4')
    learn.fit_flat_cos(2, lr=2e-04, pct_start=0)
    learn.save('Comb_6')
    learn.fit_flat_cos(2, lr=2e-04, pct_start=0)
    learn.save('Comb_8')
    learn.fit_flat_cos(2, lr=2e-04, pct_start=0)
    learn.save('Comb_10')
    learn.fit_flat_cos(2, lr=2e-04, pct_start=0)
    learn.save('Comb_12')
    
    
    # ------------
    # One more cycle with val data
    # ------------
    train, valid = get_datasets(region_ids, train_on_validation=True)
    train_dl = FastDataLoader(dataset=train,
                          bs=bs,
                          num_workers=num_workers,
                          pin_memory=True,
                          shuffle=True,
                          device=torch.device(device))
    valid_dl = FastDataLoader(dataset=valid,
                          bs=bs,
                          num_workers=num_workers,
                          pin_memory=True,
                          shuffle=True,
                          device=torch.device(device))
    data = DataLoaders(train_dl, valid_dl, device=torch.device(device))
    learn = Learner(
        data, 
        Model.to(device), 
        loss_func=VUNetLoss2, metrics=[valid_leaderboard, valid_leaderboard2], 
        model_dir=folder_to_save_models, cbs=CSVLogger(append=True))
    learn.fit_flat_cos(2, lr=2e-04, pct_start=0)
    learn.save('Lomb_14')
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r", "--regions", nargs='+', required=False, default=['R1', 'R2', 'R3'], 
                        help="The regions to train on (default is R1 R2 R3)")
    parser.add_argument("-m", "--weights_dir", type=str, required=False, default='weights', 
                        help="Location to save weights")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=12, 
                        help="The batch size")
    parser.add_argument("-w", "--num_workers", type=int, required=False, default=3, 
                        help="Number of workers for dataloader")
    parser.add_argument("-g", "--gpu", type=str, required=False, default='cuda', 
                        help="The device 'cuda' for gpu or 'cpu' for cpu")

    return parser

def main():
    
    parser = set_parser()
    options = parser.parse_args()

    train(options.regions, options.weights_dir, options.batch_size, options.num_workers, options.gpu)

if __name__ == "__main__":
    main()

    """ examples of usage:

        - Training

    python weather4cast/traing.py -r R1 R2 R3 -m weights -b 12 -w 3 -g 'cuda'

    """