# Author: Pedro Herruzo
# Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import pathlib
import sys
import os
import json

#module_dir = str(pathlib.Path(os.getcwd()).parent)
#sys.path.append(module_dir)

from utils.w4c_dataloader_aio import create_dataset

import utils.config as cf
import utils.data_utils_aio as data_utils
from utils.vunet_model import Net1
import torch

import glob

# ------------
# 1. Create output folders
# ------------
def create_directory_structure(root, region, folder_name='inference'):
    """
    Create the inference output directory structure at given root path: root/folder_name 
    """

    # create the main fo
    metadata_path = os.path.join(root, folder_name)
    out_path = os.path.join(metadata_path, region, 'test')
    try:
        # os.makedirs(r_path)
        os.makedirs(out_path)
        print(f'created path: {out_path}')
        
    except:
        print(f'failed to create directory structure, maybe they already exist: {out_path}')
    return metadata_path, out_path

# ------------
# 2. Prepare metadata needed by the weather4cast dataloader: `w4c_dataloader`
# ------------
def get_bin_labels():
    return ['{}{}{}{}00'.format('0'*bool(i<10), i, '0'*bool(j<10), j) for i in np.arange(0, 24, 1) for j in np.arange(0, 60, 15)]

def fn_time_2_timebin():
    times = get_bin_labels()
    bins = {t_str: tbin for tbin, t_str in enumerate(times) }
    return bins

def get_out_bins(start, end, id_date, n_bins, time_bin_labels=get_bin_labels()):
    """ Creates the meta-data for the time intervals to be predicted """
    i = 0
    bins_holder = {}
    
    for idx_bin in range(start, end):
        
        if  idx_bin%n_bins == 0: # jump to next day
            day = int(id_date[-3:]) + 1 # ToDo %365 
            
            zeros_before = '0'*(3 - len(str(day))%4)
            id_day =  zeros_before + str(day)
            id_date = id_date[:-3]+id_day
        
        bins_holder[i] = {'id_day': id_date[-3:], 'id_bin': idx_bin%n_bins, 'time_bin': time_bin_labels[idx_bin%n_bins],
                          'date': datetime.datetime.strptime(id_date[:-3]+' '+id_date[-3:], '%Y %j').strftime('%Y%m%d')}
        i += 1
    
    return bins_holder

def create_test_csv_json(data_p, region_id, metadata_path, product='CMA', 
                         n_bins=96, n_preds=32, n_files=4):
    """ Creates a metadata filling input/output time intervals for a given folder. 
        It uses the files inside the folder of product `product` to inform the time intervals.??
    """
    if region_id in ['R1', 'R2', 'R3', 'R7', 'R8']:
        track = 'core-w4c'
    else:
        track = 'transfer-learning-w4c'

    # 1. get the dates to make inference from
    root = os.path.join(data_p, track, region_id, 'test')
    dates = [name for name in os.listdir(root) if os.path.isdir(root)]
    dates.sort()
    
    cols = ['id_date', 'split_id',	'split', 'id_day', 'date']
    date_split = []
    date_timebins = {}
    time_2_timebin = fn_time_2_timebin()

    for date in dates:
        if date.endswith('.DS_Store'):
            os.remove(os.path.join(root, date))
            continue

        # get the 4 input time intervals & sort them
        tmp_p = os.path.join(root, date, product, '*.nc')
        files = glob.glob(tmp_p)
        files.sort()
        assert len(files) == n_files, f'Number of files must be {n_files}, check your input folders'

        # get day and time from the files
        bins_day = {'bins_in': {}, 'bins_out': {}}
        for i, f in enumerate(files):
            f = f.split('_')[-1].split('Z')[0].split('T')
            day, time = f[0], f[-1]
            idx_timebin = time_2_timebin[time]

            # data to add to the json
            tmp = {'id_day': date[-3:], 'id_bin': idx_timebin, 'time_bin': time, 'date': day}
            bins_day['bins_in'][str(i)] = tmp
            # print(tmp)
            # print(day, time)

            if i == 0:
                # data to add to the csv
                date_split.append([date, 2, 'test', date[-3:], day])
        
        idx_timebin += 1 # set the next time bin (the one to start predicting)
        bins_day['bins_out'] = get_out_bins(idx_timebin, idx_timebin+n_preds, date, n_bins)
        date_timebins[date[-3:]] = bins_day
    df = pd.DataFrame(date_split, columns=cols)

    # safe the files
    df.to_csv(os.path.join(metadata_path, 'splits.csv'))
    with open(os.path.join(metadata_path, 'test_split.json'), 'w', encoding='utf-8') as f:
        json.dump(date_timebins, f, ensure_ascii=False, indent=4)
    with open(os.path.join(metadata_path, 'blacklist.json'), 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

# ------------
# 3. load data & model
# ------------
def get_data_iterator(region_id, data_path, splits_path, data_split='test', collapse_time=True, 
                      batch_size=1, shuffle=False, num_workers=0):
    """ Creates an iterator for data in region 'region_id' for the days in `splits_path`
    """
    static_data_path=os.path.join(data_path, 'static')
    params = cf.get_params(region_id=region_id, data_path=data_path, splits_path=splits_path, static_data_path=static_data_path)
    params['data_params']['collapse_time'] = collapse_time

    ds = create_dataset(data_split, params['data_params'])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    data_splits, test_sequences = data_utils.read_splits(params['data_params']['train_splits'], params['data_params']['test_splits'])
    test_dates = data_splits[data_splits.split=='test'].id_date.sort_values().values

    return iter(dataloader), test_dates, params

def load_models_and_weights(id_region, model_location, device=None):
    """ Loads a model
    """
    #print(device)
    model = Net1(in_channels=8*4+3,out_channels=4*32)
    state = torch.load(model_location, map_location=torch.device(device) )
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model

# ------------
# 4. Make predictions
# ------------
def get_preds(model, batch, device=None):
    """ Computes the output of the model on the next iterator's batch and
        returns the prediction and the date of it.
    """
    
    in_seq, out, metadata = batch
    day_in_year = metadata['in']['day_in_year'][0][0].item()
    
    if device is not None:
        #in_seq = in_seq.cuda(device=device)
        in_seq = in_seq.to(device)
    with torch.no_grad():
        y_hat = model(in_seq)
        y_hat = y_hat[0].squeeze().numpy()
    y_hat = np.clip(y_hat, 0, 1)
    #y_hat = y_hat.data.cpu().numpy()  
    
    return y_hat, day_in_year

def predictions_per_day(test_dates, model, ds_iterator, device, file_path, data_params):
    """ Computes predictions of all dates and saves them to disk """
    for target_date in test_dates:
        print(f'generating submission for date: {target_date}...')
        batch = next(ds_iterator)
        y_hat, predicted_day = get_preds(model, batch, device)
                
        # batches are sorted by date for the dataloader, that's why they coincide
        assert predicted_day==target_date, f"Error, the loaded date {predicted_day} is different than the target: {target_date}"

        f_path = os.path.join(file_path, f'{predicted_day}.h5')
        y_hat = data_utils.postprocess_fn(y_hat, data_params['target_vars'], data_params['preprocess']['source'])
        data_utils.write_data(y_hat, f_path)
        print(f'--> saved in: {f_path}')

# ------------
# 5. Main program
# ------------
def inference(data_p, region, weights, output, device):
    """ Computes predictions using inputs from the `test` folder in: `data_p/<core, transfer-learning>-w4c/region_id`
        This script must load all needed weigths from folder: `weights`
        and save predictions in folder `outputs`
    """
    # ------------
    # A. input/output preparation
    # ------------
    # 1. create a folder to save the predictions per day
    metadata_path, out_path = create_directory_structure(output, region, folder_name='inference')

    # 2. create the csv and json needed by the class `dataset` to provide single sequences per batch
    # so we can save to disk single predictions per day of shape (32, 4, 256, 256)
    create_test_csv_json(data_p, region, metadata_path)

    # ------------
    # B. model & data loading:
    #
    # This part of the code must load the data and models. If you used the same `dataset` class we provided
    # you probably only need to modify loading the models. Adapt the code so it loads the learned weights from
    # the folder `weights` you provided for them
    # ------------
    ds_iterator, test_dates, params = get_data_iterator(region, data_p, metadata_path, batch_size=1)
    model = load_models_and_weights(region, weights, device=device)

    # ------------
    # C. Predict and save the predictions
    # ------------
    #if len(models)==1:  
    predictions_per_day(test_dates, model, ds_iterator, device, out_path, params['data_params'])
    #else:
    #    predictions_per_day_ensamble(test_dates, models, ds_iterator, device, out_path, params['data_params'])
        
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data", type=str, required=True, 
        help='path to a folder containing days to be predicted (e.g. the test folder of the test dataset)')
    parser.add_argument("-r", "--region", type=str, required=False, default='R1',
        help='Region where the data belongs.')
    parser.add_argument("-w", "--weights", type=str, required=True, 
        help='path to a file containing the model weights')
    parser.add_argument("-o", "--output", type=str, required=True, 
        help='path to save the outputs of the model for each day.')
    parser.add_argument("-g", "--device", type=str, required=False, default='cpu', 
                        help="which device to use - use 'cuda' for gpu. Default is 'cpu' ")

    return parser

def main():
    
    parser = set_parser()
    options = parser.parse_args()
    #print(options.gpu)
    inference(options.data, options.region, options.weights, options.output, options.device)

if __name__ == "__main__":
    main()

    """ examples of usage:

        - inference for Region R1

    R=R1
    INPUT_PATH=data
    WEIGHTS=weights/Lomb_14.pth
    OUT_PATH=.
    python weather4cast/inference.py -d $INPUT_PATH -r $R -w $WEIGHTS -o $OUT_PATH -g 'cuda'

    """