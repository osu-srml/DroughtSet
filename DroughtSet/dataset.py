
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




class DroughtSet_SP(DroughtSet):
    def __init__(self, base_input_file, base_target_file, mask, stat, all_nan_mask, split='train'):
        super().__init__(base_input_file, base_target_file, mask, stat, split)
        
        
        full_mask = ~torch.isnan(mask)        # 替代 np.isnan
        fullrows, fullcols = torch.nonzero(full_mask, as_tuple=True)  # 替代 np.nonzero

        self.location_matrix = np.full((585, 1386), -1)
        for i, (r, c) in enumerate(zip(fullrows, fullcols)):
            self.location_matrix[r, c] = i
        
        
        self.distance_matrix = np.fromfunction(
            lambda i, j: np.sqrt((i - 2)**2 + (j - 2)**2), 
            (5, 5),
            dtype=int
        )
        self.all_nan_mask = all_nan_mask

    def get_loc(self, index):
        if self.split != 'all':
            original_index = self.list[index]
        else:
            original_index = index

        row = original_index // self.mask.shape[1]
        col = original_index % self.mask.shape[1]
        return row, col

    def __getitem__(self, index):
        if self.split != 'all':
            original_index = self.list[index]
        else:
            original_index = index

        agg_distance = torch.full((5, 5), float('inf'))
        row, col = self.get_loc(index)
        neighbors = []
        self_location = self.location_matrix[row][col]

        for i in range(row - 2, row + 3):
            for j in range(col - 2, col + 3):
                if i == row and j == col:
                    agg_distance[i - (row - 2)][j - (col - 2)] = 0.5
                    neighbors.append(self_location)
                    continue

                if i < 0 or i >= self.location_matrix.shape[0] or j < 0 or j >= self.location_matrix.shape[1]:
                    neighbors.append(self_location)
                    continue

                neighbor_loc = self.location_matrix[i][j]

                if neighbor_loc == -1 or self.all_nan_mask[neighbor_loc]:
                    neighbors.append(self_location)
                else:
                    neighbors.append(neighbor_loc)
                    agg_distance[i - (row - 2)][j - (col - 2)] = self.distance_matrix[i - (row - 2)][j - (col - 2)]

        neighbors = torch.tensor(neighbors, dtype=torch.long)

        x_cata = self.predictors_data['nlcd.mat_nlcd'][original_index]
        x_cata[torch.isnan(x_cata)] = -1

        x_num_static = torch.stack([
            (self.predictors_data[k][neighbors]) / (self.max_rec[k])
            for k in self.input_static_names
        ]).permute(1, 0)

        x_num_time = torch.stack([
            (self.predictors_data[k][neighbors]) / (self.max_rec[k])
            for k in self.input_time_names
        ]).permute(1, 0, 2, 3)

        y = torch.stack([
            (v[neighbors]) / (self.max_rec[k])
            for k, v in self.drought_indices_data.items()
        ]).permute(1, 0, 2, 3)

        return x_num_static, x_cata.to(torch.int), x_num_time, y, agg_distance, original_index

def load_with_nan_mask(datapath, batchsize):
    base_input_file = torch.load( os.path.join(datapath,"inputs.pt"))
    base_target_file = torch.load(os.path.join(datapath, 'target.pt'))
    stat =  torch.load(os.path.join(datapath, 'stat.pt')) 
    mask = torch.load(os.path.join(datapath, 'mask.pt')) 
    nan_mask = torch.load(os.path.join(datapath, 'drought_nan_mask.pt')) 
    nan_mask_list = []
    for k,v in nan_mask.items():
        nan_mask_list.append(v.flatten(start_dim=1))
        
    all_nan_mask = nan_mask_list[0].all(dim=1)
    for i in range(1,3):
        all_nan_mask = all_nan_mask & nan_mask_list[i].all(dim=1)


    train_dataset = DroughtSet_SP(base_input_file, base_target_file, mask, stat, all_nan_mask, split='train')
    test_dataset = DroughtSet_SP(base_input_file, base_target_file, mask, stat, all_nan_mask, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4, drop_last=True)
    
    
    mean_values = {'time':[], 'static':[], 'target':[]}
    for k,v in stat['mean'].items():
        if(k in train_dataset.input_time_names):
            mean_values['time'].append(v)
        if(k in train_dataset.input_static_names):
            mean_values['static'].append(v)
        if(k in train_dataset.drought_indices_names):
            mean_values['target'].append(v)
    for k,v in mean_values.items():
        mean_values[k] = torch.stack(v) 
    
    return train_dataset, test_dataset, train_loader, test_loader, mean_values, nan_mask_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--data_path', type=str, default="./Processed", help='processed data folder path')
    args = parser.parse_args()
    train_dataset, test_dataset, train_loader, test_loader = load(args.data_path, args.bs)

