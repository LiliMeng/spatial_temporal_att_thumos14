'''
load thumos 14 spatial temporal attention features, labels and image names

Author: Lili Meng
Date:  August 29th, 2018

'''
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch


class ThumosDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.dataset = pd.read_csv(csv_file)
      
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature_file = os.path.join(self.data_dir, 'features', self.dataset['Feature'][idx])
        
        label_file = os.path.join(self.data_dir, 'labels', self.dataset['Feature'][idx].replace("features", "label"))
        name_file = os.path.join(self.data_dir, 'names', self.dataset['Feature'][idx].replace("features", "name"))
        
        feature_per_video = np.load(feature_file)
        label_per_video = np.load(label_file)
        name_per_video = np.load(name_file)
    
        sample = {'feature': feature_per_video, 'label': label_per_video}
        
        return sample, list(name_per_video)


def get_loader(data_dir, csv_file, batch_size, mode='train', dataset='thumos14'):
    """Build and return data loader."""

    
    shuffle = True if mode == 'train' else False

    if dataset == 'thumos14':
        dataset = ThumosDataset(data_dir, csv_file)
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return data_loader

if __name__ == '__main__':
    
    batch_size = 30

    val_data_dir = '../../spa_features/val'
    val_csv_file = './feature_list/feature_val_list.csv'
    val_data_loader = get_loader(val_data_dir, val_csv_file, batch_size=batch_size, mode='train',
                             dataset='thumos14')
   
    for i, (val_sample, val_batch_name) in enumerate(val_data_loader):
        val_batch_feature = val_sample['feature']
        val_batch_label = val_sample['label']
        print("val_batch_feature.shape: ", val_batch_feature.shape)
        print("len(val_batch_name): ", len(val_batch_name))
        print("val_batch_label.shape: ", val_batch_label.shape)
        
        print("i: ", i)
        break
       
    test_data_dir = '../../spa_features/test'
    test_csv_file = './feature_list/feature_test_list.csv'
    test_data_loader = get_loader(test_data_dir, test_csv_file, batch_size=batch_size, mode='test',
                             dataset='thumos14')
   
    for i, (test_sample, test_batch_name) in enumerate(test_data_loader):
        test_batch_feature = test_sample['feature']
        test_batch_label = test_sample['label']
        print("test_batch_feature.shape: ", test_batch_feature.shape)
        print("len(test_batch_name): ", len(test_batch_name))
        print("test_batch_label.shape: ", test_batch_label.shape)
        
        print("i: ", i)
        break