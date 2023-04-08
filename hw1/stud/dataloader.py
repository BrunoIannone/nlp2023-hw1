import json
import torch
from torch.utils.data import Dataset
import utils


class BioDataset(Dataset):
    # TODO: vedere se transform e target_transform sono utili
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._preprocess_samples(data, labels)

    def _preprocess_samples(self, data, labels): #notebook 3
        res = []
        for sentence, label in zip(data, labels):
            res.append((sentence, label))
        # print(res)
        return(res)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.samples[index]
        return
