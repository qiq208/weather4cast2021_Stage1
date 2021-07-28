import numpy as np
from torch.utils.data import Dataset
import utils.data_utils_aio as data_utils
from utils.context_variables import get_static

class NWCSAF(Dataset):
    
    def __init__(self, data_split, products, target_vars,
                 spatial_dim, 
                 features_ordered, features_products, features_preprocess,
                 collapse_time=False, 
                 len_seq_in=4, bins_to_predict=32, day_bins=96,
                 region_id=None,  preprocess=None,
                 crop_in=None, crop_out=None,
                 extra_data='', crop_static=None, static_paths=None, return_meta=True,
                 data_path='',
                 train_splits='splits.csv', 
                 test_splits='test_split.json', 
                 black_list_path='blacklist.json', **kwargs):
        
        self.return_meta = return_meta
        self.channel_dim = 1 # specifies the dimension to concat multiple channels/variables

        # data dimensions
        self.spatial_dim = spatial_dim
        self.collapse_time = collapse_time
        self.len_seq_in = len_seq_in
        self.bins_to_predict = bins_to_predict
        self.day_bins = day_bins
        self.day_strings = ['{}{}{}{}00'.format('0'*bool(i<10), i, '0'*bool(j<10), j) for i in np.arange(0, 24, 1) for j in np.arange(0, 60, 15)]
        
        # type of data & processing variables
        self.products = products
        self.target_vars = target_vars
        self.region_id = region_id
        self.preprocess = preprocess
        self.crop_in, self.crop_out = crop_in, crop_out
        
        # load extra variables if any 
        self.extra_data, self.static_tensor, self.static_desc = [], [], []
        if extra_data != '':
            self.extra_data = extra_data.split('-')
            self.static_tensor, self.static_desc = get_static(self.extra_data, self.len_seq_in, static_paths, 
                                                              crop=crop_static, channel_dim=self.channel_dim)
        
        # data splits to load (training/validation/test)
        self.data_path = data_path + f'/{data_split}'
        self.data_split = data_split
        self.day_paths, self.test_splits = data_utils.read_splits(train_splits, test_splits)
        
    
        # prepare all elements to load - batch idx will use the object 'self.idx'
        if self.data_split != 'test':
            self.day_paths = self.day_paths[self.day_paths.split==self.data_split].reset_index()
            self.idxs = data_utils.get_double_idxs_w_blacklist(self.day_paths['id_date'].values, self.bins_to_predict, 
                                                               self.day_bins, self.len_seq_in, 
                                                               black_list_path=black_list_path)
        else:
            test_dates = self.day_paths[self.day_paths.split==self.data_split].reset_index()
            self.idxs = data_utils.get_test_duplets(test_dates['id_date'].sort_values().values, 
                                                    self.test_splits, 
                                                    self.bins_to_predict)
            self.day_paths = self.day_paths[self.day_paths.split.isin(['test', 'test-next'])].reset_index()
        
        # New Features
        self.features_ordered = features_ordered
        self.features_products = features_products
        self.features_preprocess = features_preprocess

        # is_empty and n_inp are required for fastai to work
        self.is_empty = False if (len(self) > 0) else True
        self.n_inp = 1

    def __len__(self):
        """ total number of samples (sequences of in:4-out:1 in our case) to train """
        return len(self.idxs)
    
    def load_in_seq(self, day_id, in_start_id, len_seq_in):
        """ load the input sequence """
        
        # 1. load nwcsaf products & metadata
        in_seq, in_info = data_utils.get_sequence_netcdf4(len_seq_in, in_start_id, day_id, 
                                                         self.features_products, self.data_path, self.features_ordered, 
                                                         crop=self.crop_in, preprocess=self.features_preprocess, 
                                                         day_bins=self.day_bins, 
                                                         sorted_dates=self.day_paths.id_date.sort_values().values)
        
        mask_atth = in_seq[:,0,:,:]>0
        in_seq = in_seq.reshape(-1, in_seq.shape[-2], in_seq.shape[-1])
        in_seq = np.concatenate((in_seq, self.static_tensor[0, -3:, ...]), axis=0)
        in_info['channels'] += self.static_desc
        in_seq = np.concatenate((in_seq, mask_atth), axis=0)
        in_info['channels'] += ['atth_mask']
        ## 2. Load extra features
        #if len(self.static_tensor)!=0: # 2.1 static features
        #    in_seq = np.concatenate((in_seq, self.static_tensor), axis=self.channel_dim)
        #    in_info['channels'] += self.static_desc
        
        # 4. Collapse time if needed and set the appropriate data type for learning
        #if self.collapse_time:
        #    in_seq = data_utils.time_2_channels(in_seq, *self.spatial_dim)
        
        in_seq = in_seq.astype(np.float32)
        
        return in_seq, in_info        
    
    def load_out_seq(self, day_id, in_start_id, len_seq_in):
        """ load the output sequence """
        
        # 1. load nwcsaf products & metadata
        in_seq, in_info = data_utils.get_sequence_netcdf4(len_seq_in, in_start_id, day_id, 
                                                         self.products, self.data_path, self.target_vars, 
                                                         crop=self.crop_in, preprocess=self.preprocess['source'], 
                                                         day_bins=self.day_bins, 
                                                         sorted_dates=self.day_paths.id_date.sort_values().values)
        
        ## 2. Load extra features
        #if len(self.static_tensor)!=0: # 2.1 static features
        #    in_seq = np.concatenate((in_seq, self.static_tensor), axis=self.channel_dim)
        #    in_info['channels'] += self.static_desc
        
        # 4. Collapse time if needed and set the appropriate data type for learning
        if self.collapse_time:
            in_seq = data_utils.time_2_channels(in_seq, *self.spatial_dim)
        
        in_seq = in_seq.astype(np.float32)
        
        return in_seq, in_info   

    def load_in_out(self, day_id, in_start_id):
        """ load input/output data """
        
        # load input sequence
        in_seq, in_info = self.load_in_seq(day_id, in_start_id, self.len_seq_in)
        
        # 2. Load extra features
        #if len(self.static_tensor)!=0: # 2.1 static features
        #    in_seq = np.concatenate((in_seq, self.static_tensor), axis=self.channel_dim)
        #    in_info['channels'] += self.static_desc
        
        # load ground truth
        if self.data_split != 'test':
            
            # load output sequence
            out_seq, out_info = self.load_out_seq(day_id, in_start_id + self.len_seq_in, self.bins_to_predict)

            metadata = {'in': in_info, 
                        'out': out_info}
        else:
            out_seq = np.asarray([]) # we don't have the ground truth for the test split
            metadata = {'in': in_info, 
                        'out': {'day_in_year': [day_id],'masks': in_info['masks']}}
        out_seq = out_seq.astype(np.float32)
        #print('a')
        if self.return_meta:
            return in_seq, out_seq, metadata
        return in_seq, out_seq
    
    def __getitem__(self, idx):
        """ load 1 sequence (1 sample) """
        day_id, in_start_id = self.idxs[idx]
        return self.load_in_out(day_id, in_start_id)
    
    def get_date(self, id_day):
        """ get date from day_in_year id """
        return str(self.day_paths[self.day_paths.id_date==id_day]['date'].values[0])
    
    def geti(self, idx=0):
        """ this function allows you to get 1 sample for debugging
            Note that the batch dimension is missing, so it is added
            
            example: 
                ds = create_dataset(data_split, params)
                in_seq, out, metadata = ds.geti(0)
        """
        in_seq, out, metadata = self.__getitem__(idx)
        in_seq = np.expand_dims(in_seq, axis=0)
        out = np.expand_dims(out, axis=0)
        metadata = np.expand_dims(metadata, axis=0)

        return in_seq, out, metadata
    
def create_dataset(data_split, params):
    return NWCSAF(data_split, **params)