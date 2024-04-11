"""
Author@Jingyuan Oct 28th first version, updated on Nov 1st
This file is the demo to combine sessions and get the samples for training, testing and validation
Also the file includes channel ids which map each array element to the physical channel id.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch 
from torch.utils.data import Dataset, DataLoader 


class Monkey(Dataset):
    def __init__(self, filename, split='train') -> None:
        super().__init__()
        """
        filenames: choice from beignet or affi to load corresponding dataset
        return:
        training_data: samples to train with shape (num_samples, num_timestep, num_channels, num_bands)
        test_data: samples to test with shape (num_samples, num_timestep, num_channels, num_bands)
        val_data: samples to train with shape (num_samples, num_timestep, num_channels, num_bands)
        """
        if filename =='affi' or filename == 'beignet':
            lfp_array = np.load(f'./storage/data/lfp_{filename}.npy')
            indices = np.load(f'./storage/data/tvts_{filename}_split.npz')
            if split == 'train': 
                index = indices['train_index']
            elif split == 'val': 
                index = indices['val_index']
            else: 
                index = indices['testing_index']
            
            
            data = lfp_array[index]
            print(f'Total number of {split} samples', len(data), 'Number of Channels:', data.shape[2])
        else:
            raise NotImplementedError('No such a dataset')
        std_times_4 = np.expand_dims(4 * np.std(data.reshape(1, 1, data.shape[2], -1), axis=-1), -1)
        mean = np.expand_dims(np.mean(data.reshape(1, 1, data.shape[2], -1), axis=-1), -1)
        
        self.data = torch.FloatTensor((data - mean) / std_times_4)
        self.length = data.shape[0]

    @staticmethod
    def calculate_corr_matrix(data):
        data = np.swapaxes(data, 0, 2)
        data = data.reshape(data.shape[0], -1)
        corr = np.corrcoef(data)
        return corr 
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        x = self.data[index, :, :]
        return x 
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='beignet')
    args = parser.parse_args()
    dataset1 = Monkey(args.dataset, split='train')
    dataset2 = Monkey(args.dataset, split='val')
    dataset3 = Monkey(args.dataset, split='test')
    print(torch.min(dataset1.data), torch.max(dataset1.data))

    # corr = Monkey.calculate_corr_matrix(dataset.data)
    # train_loader = DataLoader(dataset, batch_size=32)
    # batch = next(iter(train_loader))
    # print(batch[:,0,:,:].shape)
    # corr = calculate_corr_matrix(training_data) # initial value of adjacency matrix 

