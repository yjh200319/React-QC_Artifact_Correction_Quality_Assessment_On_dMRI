import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from os.path import join


def z_score(volume):
    std = np.std(volume)
    mean = np.mean(volume)
    return (volume - mean) / std


class Train_H5Dataset(Dataset):
    def __init__(self, h5_file_path, artifact_type, train_name_list, transform=None):
        self.h5_file_path = h5_file_path
        self.artifact_type = artifact_type
        self.transform = transform
        self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
        self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
        self.mask_h5_path = join(h5_file_path, f"mask.h5")

        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            all_keys = list(h5_file.keys())
            self.dataset_names = [key for key in all_keys if any(sub in key for sub in train_name_list)]

    def __len__(self):
        return len(self.dataset_names)

    def __getitem__(self, idx):
        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            artifact_data = h5_file[dataset_name][:]

        with h5py.File(self.ground_truth_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            ground_truth_data = h5_file[dataset_name][:]

        sub = dataset_name.split('_')[0]
        # sub1 = dataset_name.split('_')[1]
        # sub = sub+'_'+sub1
        with h5py.File(self.mask_h5_path, 'r') as h5_file:
            mask = h5_file[sub][:]

        # 将数据转换为单通道3D张量
        slice_art_data = torch.tensor(z_score(artifact_data * mask), dtype=torch.float32).unsqueeze(0)
        slice_gt_data = torch.tensor(z_score(ground_truth_data * mask), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return slice_art_data, slice_gt_data, mask

class Val_H5Dataset(Dataset):
    def __init__(self, h5_file_path, artifact_type, val_name_list, transform=None):
        self.h5_file_path = h5_file_path
        self.artifact_type = artifact_type
        self.transform = transform
        self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
        self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
        self.mask_h5_path = join(h5_file_path, f"mask.h5")

        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            all_keys = list(h5_file.keys())
            self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]

    def __len__(self):
        return len(self.dataset_names)

    def __getitem__(self, idx):
        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            artifact_data = h5_file[dataset_name][:]

        with h5py.File(self.ground_truth_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            ground_truth_data = h5_file[dataset_name][:]

        sub = dataset_name.split('_')[0]
        # sub1 = dataset_name.split('_')[1]
        # sub = sub+'_'+sub1
        with h5py.File(self.mask_h5_path, 'r') as h5_file:
            mask = h5_file[sub][:]

        # 将数据转换为单通道3D张量
        slice_art_data = torch.tensor(z_score(artifact_data * mask), dtype=torch.float32).unsqueeze(0)
        slice_gt_data = torch.tensor(z_score(ground_truth_data * mask), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return slice_art_data, slice_gt_data, mask



class Test_H5Dataset(Dataset):
    def __init__(self, h5_file_path, artifact_type, test_name_list, transform=None):
        self.h5_file_path = h5_file_path
        self.artifact_type = artifact_type
        self.transform = transform
        self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
        self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
        self.mask_h5_path = join(h5_file_path, f"mask.h5")

        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            all_keys = list(h5_file.keys())
            self.dataset_names = [key for key in all_keys if any(sub in key for sub in test_name_list)]

    def __len__(self):
        return len(self.dataset_names)

    def __getitem__(self, idx):
        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            artifact_data = h5_file[dataset_name][:]

        with h5py.File(self.ground_truth_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            ground_truth_data = h5_file[dataset_name][:]

        sub = dataset_name.split('_')[0]
        # sub1 = dataset_name.split('_')[1]
        # sub = sub+'_'+sub1
        with h5py.File(self.mask_h5_path, 'r') as h5_file:
            mask = h5_file[sub][:]

        # 将数据转换为单通道3D张量
        slice_art_data = torch.tensor(z_score(artifact_data * mask), dtype=torch.float32).unsqueeze(0)
        slice_gt_data = torch.tensor(z_score(ground_truth_data * mask), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return slice_art_data, slice_gt_data, mask


if __name__ == '__main__':
    # data_dir = r'E:/NIMG_small_data/Caffine/Train'
    h5_file_path_1 = r'/data/Dataset1/Train/spike.h5'
