
import argparse
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import mat73
import os

class DroughtSet(Dataset):
    def __init__(self, base_input_file, base_target_file, mask, stat, split='train'):
        
        self.drought_indices_data = base_target_file
        self.drought_indices_names = []
        self.max_rec = stat['max']
        self.min_rec = stat['min']
        for k, v in self.drought_indices_data.items():
            self.drought_indices_names.append(k)
        
        self.predictors_data = base_input_file
        self.input_cata_names,self.input_time_names, self.input_static_names= [],[],[]
        for k, v in self.predictors_data.items():
            if('nlcd' in k):
                self.input_cata_names.append(k)
                continue
            if len(v.shape) >= 3:
                self.input_time_names.append(k)
            if len(v.shape) <= 2:
                self.input_static_names.append(k)
        
        print(f"time varing attributes: {self.input_time_names}")
        print(f"static attributes: {self.input_static_names}")
        print(f"cata attributes: {self.input_cata_names}")
            
        self.split = split
        indices = np.arange(mask.size(0) * mask.size(1)).reshape(mask.shape)
        train_list = indices[mask == 1.0].flatten()
        test_list = indices[mask == 2.0].flatten()
        all_indices = np.concatenate((train_list, test_list))
        reordered_indices = {original: new for new, original in enumerate(sorted(all_indices))}
        train_list = np.array([reordered_indices[idx] for idx in train_list], dtype=int)
        test_list = np.array([reordered_indices[idx] for idx in test_list], dtype=int)
        
        if(split=='train'):
            self.mask = mask == 1.0
            self.list = train_list.astype(int)
        elif(split=='test'):
            self.mask = mask == 2.0
            self.list = test_list.astype(int)
        elif(split=='all'):
            self.mask = ~np.isnan(mask)

    def __getitem__(self, index):
        if(self.split != 'all'):
            original_index = self.list[index]
        else:
            original_index = index
        
        x_cata = self.predictors_data['nlcd.mat_nlcd'][original_index]
        x_cata[torch.isnan(x_cata)] = -1
        x_num_static = torch.stack([(self.predictors_data[k][original_index]) / (self.max_rec[k]) for k in self.input_static_names])
        x_num_dyna = torch.stack([(self.predictors_data[k][original_index]) / (self.max_rec[k]) for k in self.input_time_names])
        y = torch.stack([(v[original_index]) / (self.max_rec[k])  for k,v in self.drought_indices_data.items()])
        return x_num_static, x_cata.to(torch.int), x_num_dyna, y
    
    def __len__(self):
        return self.mask.sum()

        
def load(datapath, batchsize):
    base_input_file = torch.load( os.path.join(datapath,"inputs.pt"))
    base_target_file = torch.load(os.path.join(datapath, 'target.pt'))
    stat =  torch.load(os.path.join(datapath, 'stat.pt')) 
    mask = torch.load(os.path.join(datapath, 'mask.pt')) 


    train_dataset = DroughtSet(base_input_file, base_target_file, mask, stat, split='train')
    test_dataset = DroughtSet(base_input_file, base_target_file, mask, stat, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4, drop_last=True)
    
    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--data_path', type=str, default="./Processed", help='processed data folder path')
    args = parser.parse_args()
    train_dataset, test_dataset, train_loader, test_loader = load(args.data_path, args.bs)

