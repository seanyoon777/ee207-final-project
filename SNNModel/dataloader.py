import h5py
import numpy as np
import torch
from torch.utils.data import Subset
from neurobench.datasets import PrimateReaching

data_path = './data/indy_20160630_01.mat'


class Dataloader(object):

    def __init__(self):

        pr_dataset = PrimateReaching(file_path='data',
                                     filename="indy_20160630_01.mat",
                                     num_steps=1,
                                     train_ratio=0.5,
                                     bin_width=0.004,
                                     biological_delay=0,
                                     download=False)

        print("Loaded pr dataset")


# TODO: are spikes available for different samples?
        with h5py.File(data_path, 'r') as data:
            # Extract the data
            timesteps = torch.tensor(np.array(data['t'])).flatten()  # Ensure t is a 1D array
            finger_pos = torch.tensor(np.array(data['finger_pos']).T)  # Ensure proper shape and convert to PyTorch tensor

            # Determine the number of samples (n) and units (k)
            n = len(timesteps)
            k = len(data['spikes'][0])

            # Initialize the nxk array with zeros
            spike_matrix = np.zeros((n, k), dtype=np.float32)

            # Populate the spike_matrix while reading the spike data
            for unit_idx, channel_ref in enumerate(data['spikes'][0]):
                channel = data[channel_ref]  # Dereference the channel
                for unit_ref in channel:
                    unit_spike_times = np.array(
                        unit_ref).flatten()  # Dereference the unit spike times and flatten

                    spike_indices = np.searchsorted(timesteps, unit_spike_times)
                    spike_indices = spike_indices[spike_indices < n]  # Ensure indices are within bounds
                    spike_matrix[spike_indices, unit_idx] = 1

        train_kin_data = finger_pos[pr_dataset.ind_train]
        test_kin_data = finger_pos[pr_dataset.ind_test]

        train_neural_data = spike_matrix[pr_dataset.ind_train]
        test_neural_data = spike_matrix[pr_dataset.ind_test]


        # Load and normalize the data
        self.trainY = np.transpose(train_kin_data.numpy())
        t_mean = np.mean(self.trainY, axis=1, keepdims=True)
        t_std = np.std(self.trainY, axis=1, keepdims=True)

        self.trainY = np.transpose((self.trainY - t_mean) / t_std)

        self.testY = np.transpose(test_kin_data.numpy())
        self.testY = np.transpose((self.testY - t_mean) / t_std)

        self.trainX = train_neural_data
        self.testX = test_neural_data

    def getTrainData(self, type):
        if type == 'KinData':
            return self.trainY
        elif type == 'NeuralData':
            return self.trainX

    def getTestData(self, type):
        if type == 'KinData':
            return self.testY
        elif type == 'NeuralData':
            return self.testX

    def getData(self):
        return self.trainX, self.trainY, self.testX, self.testY
