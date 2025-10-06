import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UnsupervisedTrainDataset(Dataset):
    def __init__(self, data_dir, config):
        self.v_feat = torch.from_numpy(np.load(os.path.join(data_dir, 'train_video_features.npy'))).float()
        self.s_feat = torch.from_numpy(np.load(os.path.join(data_dir, 'train_speech_features.npy'))).float()
        self.t_feat = torch.from_numpy(np.load(os.path.join(data_dir, 'train_text_features.npy'))).float()
        self.config = config
        print(f"Loaded UNSUPERVISED training dataset with {len(self.v_feat)} samples.")

    def __len__(self):
        return len(self.v_feat)

    def __getitem__(self, index):
        return self.v_feat[index], self.s_feat[index], self.t_feat[index]

class SupervisedDataset(Dataset):
    def __init__(self, data_dir, split, config):
        self.v_feat = torch.from_numpy(np.load(os.path.join(data_dir, f'{split}_video_features.npy'))).float()
        self.s_feat = torch.from_numpy(np.load(os.path.join(data_dir, f'{split}_speech_features.npy'))).float()
        self.t_feat = torch.from_numpy(np.load(os.path.join(data_dir, f'{split}_text_features.npy'))).float()
        self.labels = torch.from_numpy(np.load(os.path.join(data_dir, f'{split}_label_{config.n_classes}_class.npy'))).long()
        print(f"Loaded SUPERVISED {split} dataset with {len(self.v_feat)} samples.")

    def __len__(self):
        return len(self.v_feat)

    def __getitem__(self, index):
        return self.v_feat[index], self.s_feat[index], self.t_feat[index], self.labels[index]
