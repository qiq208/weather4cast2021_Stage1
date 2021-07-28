import numpy as np
import os

def prepare_crop(regions, region_id):
    """ this function prepares the expected parameters to crop images per region
        e.g., to crop latitudes to the region of interest 
    """
    x, y = regions[region_id]['up_left']
    crop = {'x_start': x, 'y_start': y, 'size': regions[region_id]['size']}
    return crop

def n_extra_vars(string_vars):
    """ computes how many extra variables will be used """
    if string_vars=='':
        len_extra = 0
    else:
        len_extra = len(string_vars.split('-'))
        if 'l' in string_vars: 
            len_extra += 1 # 'l' loads both lat/lon, so 2 vars (not 1)
    return len_extra

def get_prod_name(product):
    """ get the folder name for each product. Note that only the folder containing ASII 
        have a slightly different name 
    """
    if product=='ASII':
        product = 'ASII-TF'
    return product

def get_params(region_id='R1', 
               data_path=os.path.join(os.getcwd(), '../data'),
               splits_path=os.path.join(os.getcwd(), 'utils'),
               static_data_path=os.path.join(os.getcwd(), '../data/static'),
               size=256,
               collapse_time=False):
    """ Set paths & parameters to load/transform/save data and models.

    Args:
        region_id (str, optional): Region to load data from]. Defaults to 'R1'.
        data_path (str, optional): path to the parent folder containing folders 
            for the core competition (*/w4c-core-stage-1) and/or 
            transfer learning comptition (*/w4c-transfer-learning-stage-1'). 
            Defaults to 'data'.
        splits_path (str, optional): Path to the folder containing the csv and json files defining 
            the data splits. 
            Defaults to 'utils'.
        static_data_path (str, optional): Path to the folder containing the static channels. 
            Defaults to 'data/static'.
        size (int, optional): Size of the region. Default to 256.

    Returns:
        dict: Contains the params
    """

    data_params = {}
    model_params = {}
    training_params = {}
    optimization_params = {}

    regions = {'R3': {'up_left': (935, 400), 'split': 'train', 'desc': 'South West\nEurope', 'size': size}, 
               'R6': {'up_left': (1270, 250), 'split': 'test', 'desc': 'Central\nEurope', 'size': size}, 
               'R2': {'up_left': (1550, 200), 'split': 'train', 'desc': 'Eastern\nEurope', 'size': size},  
               'R1': {'up_left': (1850, 760), 'split': 'train', 'desc': 'Nile Region', 'size': size}, 
               'R5': {'up_left': (1300, 550), 'split': 'test', 'desc': 'South\nMediterranean', 'size': size}, 
               'R4': {'up_left': (1020, 670), 'split': 'test', 'desc': 'Central\nMaghreb', 'size': size},
               'R7': {'up_left': (1700, 470), 'split': 'train', 'desc': 'Bosphorus', 'size': size}, 
               'R8': {'up_left': (750, 670), 'split': 'train', 'desc': 'East\nMaghreb', 'size': size}, 
               'R9': {'up_left': (450, 760), 'split': 'test', 'desc': 'Canarian Islands', 'size': size}, 
               'R10': {'up_left': (250, 500), 'split': 'test', 'desc': 'Azores Islands', 'size': size}, 
               'R11': {'up_left': (1000, 130), 'split': 'test', 'desc': 'North West\nEurope','size': size}
               } 
    print(f'Using data for region {region_id} | size: {size} | {regions[region_id]["desc"]}')

    # ------------
    # 1. Files to load
    # ------------
    if region_id in ['R1', 'R2', 'R3', 'R7', 'R8']:
        track = 'core-w4c'
    else:
        track = 'transfer-learning-w4c'

    data_params['data_path'] = os.path.join(data_path, track, region_id)
    
    data_params['static_paths'] = {}
    data_params['static_paths']['l'] = os.path.join(static_data_path, 'Navigation_of_S_NWC_CT_MSG4_Europe-VISIR_20201106T120000Z.nc')
    data_params['static_paths']['e'] = os.path.join(static_data_path, 'S_NWC_TOPO_MSG4_+000.0_Europe-VISIR.raw')

    data_params['train_splits'] = os.path.join(splits_path, 'splits.csv')
    data_params['test_splits'] = os.path.join(splits_path, 'test_split.json')
    data_params['black_list_path'] = os.path.join(splits_path, 'blacklist.json')
    

    # ------------
    # 2. Data params    
    # ------------
    data_params['collapse_time'] = collapse_time
    data_params['extra_data'] = 'l-e' # use '' to not use static features
    data_params['target_vars'] = ['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma']
    data_params['products'] = {'CTTH': ['temperature'], 
                               'CRR': ['crr_intensity'], 
                               'ASII': ['asii_turb_trop_prob'], 
                               'CMA': ['cma']}
    data_params['weigths'] = {'temperature': .25, 
                               'crr_intensity': .25, 
                               'asii_turb_trop_prob': .25, 
                               'cma': .25} # to use by the metric

    data_params['depth'] = len(data_params['target_vars']) + n_extra_vars(data_params['extra_data']) + 1 # lead time is added
    data_params['spatial_dim'] = (size, size)
    data_params['crop_static'] = prepare_crop(regions, region_id)
    data_params['crop_in'] = None
    data_params['crop_out'] = None
    data_params['train_region_id'] = region_id+'_mse'*1 # this is actually used by the model, not the data ??????
    data_params['region_id'] = region_id
    data_params['len_seq_in'] = 4  # time-bins of 15 minutes
    data_params['len_seq_out'] = 1 # time-bins
    data_params['bins_to_predict'] = 8*4 # hours x (time-bins per hour =4) # not used
    data_params['day_bins'] = 96
    data_params['seq_mode'] = 'sliding_window' # not used
    data_params['width'] = 256 # not used
    data_params['height'] = 256 # not used

    # preprocessing:
    #    a. fill_value: value to replace NaNs (currently temperature is the one that has more)
    #    b. max_value: maximum value of the variable when it's saved on disk as integer
    #    c. scale_factor: netCDF automatically uses this value to re-scale the value
    #    d. add_offset: netCDF automatically uses this value to shift a variable
    #
    # c. and d. together mean that once loaded, the data is in the scale [add_offset, max_value*scale_factor + add_offset]
    # Hence, to normalize the data between [0, 1] we must use:
    #    data = (data-add_offset)/(max_value*scale_factor - add_offset)
    preprocess = {'cma':                 {'fill_value': 0, 'max_value': 1, 'add_offset': 0, 'scale_factor': 1}, 
                  'temperature':         {'fill_value': 0, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)}, 
                  'crr_intensity':       {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
                  'asii_turb_trop_prob': {'fill_value': 0, 'max_value': 100, 'add_offset': 0, 'scale_factor': 1}}
    preprocess_tgt = {'cma':                 {'fill_value': np.nan, 'max_value': 1, 'add_offset': 0, 'scale_factor': 1}, 
                  'temperature':         {'fill_value': np.nan, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)}, 
                  'crr_intensity':       {'fill_value': np.nan, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
                  'asii_turb_trop_prob': {'fill_value': np.nan, 'max_value': 100, 'add_offset': 0, 'scale_factor': 1}}
    data_params['preprocess'] = {'source': preprocess, 'target': preprocess_tgt}
    
    data_params['features_ordered'] = ['temperature', 'ctth_pres', 'crr_intensity', 'crr_accum', 'asii_turb_trop_prob', 'ct', 'cma']
    data_params['features_products'] = {'CTTH': ['temperature', 'ctth_pres'], 
                                        'CRR': ['crr_intensity', 'crr_accum'], 
                                        'ASII': ['asii_turb_trop_prob'], 
                                        'CMA': ['cma'],
                                        'CT': ['ct']}
    data_params['features_preprocess'] = {'ct':                 {'fill_value': 0, 'max_value': 15, 'add_offset': 0, 'scale_factor': 1},
     'temperature':         {'fill_value': 0, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)},
     'ctth_pres':         {'fill_value': 0, 'max_value': 11000, 'add_offset': 0, 'scale_factor': np.float32(10.0)},
     #'ctth_alti':         {'fill_value': 0, 'max_value': 25000, 'add_offset': -2000.0, 'scale_factor': np.float32(1.0)},
     'crr_intensity':       {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
     'crr_accum':       {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
     'asii_turb_trop_prob': {'fill_value': 0, 'max_value': 100, 'add_offset': 0, 'scale_factor': 1},
     'cma':                 {'fill_value': 0, 'max_value': 1, 'add_offset': 0, 'scale_factor': 1}}
    # ------------
    # 3. Model params
    # ------------
    if data_params['collapse_time']:
        model_params['in_channels'] = data_params['depth'] * data_params['len_seq_in']
    else:
        model_params['in_channels'] = data_params['depth']
    model_params['n_classes'] = len(data_params['target_vars'])
    model_params['depth'] = 5
    model_params['wf'] = 6
    model_params['padding'] = True
    model_params['batch_norm'] = False
    model_params['up_mode'] = 'upconv'

    # ------------
    # 4. Training params
    # ------------
    training_params['batch_size'] = 64
    training_params['n_workers'] = 8

    params = {
        'data_params': data_params,
        'model_params': model_params,
        'training_params': training_params,
        'optimization_params': optimization_params,
    }
    
    return params

if __name__ == '__main__':
    # this is only executed when the module is run directly.
    print(get_params())
