import h5py
import numpy as np
import torch
from torch.utils.data import Subset
from neurobench.datasets import PrimateReaching

data_path = './data/indy_20160630_01.mat'


class Dataloader:
    def __init__(self, spike_units='all', normalize=True, center_zero=True, aggreagate_chs=True, bin_size=1, bad_ch_cutoff=1):
        """_summary_

        Args:
            spike_units: string of list of spike unit numbers to use (e.g. [0, 2, 4])
            normalize (bool, optional): _description_. Defaults to True.
            bin_size (int, optional): _description_. Defaults to 1.
            ex_cutoff : excludes spiking units that spike less than this cutoff value 
        """
        self.pr_dataset = PrimateReaching(file_path='data',
                                          filename="indy_20160630_01.mat",
                                          num_steps=1,
                                          train_ratio=0.5,
                                          bin_width=0.004,
                                          biological_delay=0,
                                          download=False)
        print("Loaded pr dataset")
        
        self.spike_units   = spike_units 
        self.normalize     = normalize 
        self.center_zero   = center_zero 
        self.aggregate_chs = aggreagate_chs
        self.bin_size      = bin_size
        self.bad_ch_cutoff = bad_ch_cutoff
        
        with h5py.File(data_path, 'r') as data:
            # initialize spike units to use 
            if spike_units == 'all': 
                n_sp = data['spikes'].shape[0]
                self.spike_units = [i for i in range(n_sp)]
            
            # get spike matrix and split into train / test set 
            spike_matrix = self._get_spike_mat(data)
            trainX, testX, train_finger_pos, test_finger_pos, train_cursor_pos, test_cursor_pos = self._train_test_split(data, spike_matrix)
            
            # bin data
            if self.bin_size > 1: 
                trainX = self._bin_data(trainX)
                testX  = self._bin_data(testX)

            # convert positions to velocity 
            trainY = self._pos2vel(train_finger_pos)
            testY = self._pos2vel(test_finger_pos)
            trainY_cursor = self._pos2vel(train_cursor_pos)
            testY_cursor = self._pos2vel(test_cursor_pos)            
            print(f'Binned into {4*self.bin_size}ms intervals')
            
            # normalize velocities if necessary 
            if self.normalize: 
                trainX, testX = self._normalize(trainX, testX)
            if self.center_zero: 
                trainY, testY = self._center_zero(trainY, testY)

            # exclude channels that are too sparse 
            trainX, testX = self._exclude_channels(trainX, testX)
            
            self.data = trainX, testX, trainY, testY, trainY_cursor, testY_cursor
            

    def _get_spike_mat(self, data): 
        # extract timesteps and spike channel params
        timesteps = torch.tensor(np.array(data['t'])).flatten()  # Ensure t is a 1D array
        n_ch = len(data['spikes'][0])  # number of channels 
        n_sp = len(self.spike_units)   # number of spike units to use for each channel
        
        # initialize spike matrix with zeros 
        n = len(timesteps)
        k = n_sp * n_ch 
        spike_matrix = np.zeros((n, k), dtype=np.float32)
        
        # read spike data and populate spike matrix 
        for sp_unit in self.spike_units: 
            for unit_idx, channel_ref in enumerate(data['spikes'][sp_unit]): 
                channel = data[channel_ref]  # Dereference the channel
                for unit_ref in channel: 
                    unit_spike_times = np.array(unit_ref).flatten()  # Dereference the unit spike times and flatten
                    spike_indices    = np.searchsorted(timesteps, unit_spike_times)
                    spike_indices    = spike_indices[spike_indices < n]  # Ensure indices are within bounds
                    spike_matrix[spike_indices, sp_unit*n_ch+unit_idx] = 1
                    
        return spike_matrix
    
    
    def _train_test_split(self, data, spike_matrix):
        finger_pos = torch.tensor(np.array(data['finger_pos']).T)  # Ensure proper shape and convert to PyTorch tensor
        cursor_pos = torch.tensor(np.array(data['cursor_pos']).T)  # Ensure proper shape and convert to PyTorch tensor
        # get train and test indices
        train_idx = self.pr_dataset.ind_train
        test_idx  = self.pr_dataset.ind_test
        
        train_neural = spike_matrix[train_idx]
        test_neural  = spike_matrix[test_idx]
        
        train_finger_pos = finger_pos[train_idx][:,:3]  # extract z, -x, -y positions and drop spherical coords
        test_finger_pos  = finger_pos[test_idx][:,:3]   # extract z, -x, -y positions and drop spherical coords
        
        train_cursor_pos = cursor_pos[train_idx]
        test_cursor_pos  = cursor_pos[test_idx]
        
        return train_neural, test_neural, train_finger_pos, test_finger_pos, train_cursor_pos, test_cursor_pos
        
        
    def _bin_data(self, neural): 
        n_samples, n_spikes = neural.shape
        n_bins = int(np.ceil(n_samples / self.bin_size))
        
        X = np.zeros(shape=(n_bins, n_spikes), dtype=np.float32)
        for bin, i in enumerate(range(0, n_samples, self.bin_size)): 
            X[bin] = np.mean(neural[i:i+self.bin_size], axis=0)
        
        return X


    def _pos2vel(self, y): 
        n_samples, n_outputs = y.shape
        n_bins = int(np.ceil(n_samples / self.bin_size))
        
        y_pos = np.zeros(shape=(n_bins, n_outputs), dtype=np.float32)
        y_vel = np.zeros(shape=(n_bins, n_outputs), dtype=np.float32)
        for bin, i in enumerate(range(0, n_samples, self.bin_size)): 
            binned_y = y[i:i+self.bin_size]
            y_vel[bin] = binned_y[-1] - binned_y[0]
            y_pos[bin] = binned_y[-1]
        
        y_out = np.vstack([y_pos, y_vel])
        return y_out
        
    
    def _normalize(self, trainX, testX): 
        mean = np.mean(trainX, axis=0, keepdims=True)
        std  = np.std(trainX, axis=0, keepdims=True)
        
        return (trainX - mean) / std, (testX - mean) / std
    
    
    def _center_zero(self, trainY, testY): 
        print(trainY.shape)
        mean = np.mean(trainY, axis=0, keepdims=True)
        self.finger_mean_vel = mean 
        return trainY - mean, testY - mean
        
    
    def _exclude_channels(self, trainX, testX): 
        train_chs = np.sum(trainX, axis=0) > self.bad_ch_cutoff
        test_chs  = np.sum(testX, axis=0) > self.bad_ch_cutoff
        all_chs   = np.logical_and(train_chs, test_chs)
        
        return trainX[:,all_chs], testX[:,all_chs]
    
    
    def __call__(self): 
        return self.data
