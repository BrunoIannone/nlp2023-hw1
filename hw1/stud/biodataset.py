import json
import torch
from torch.utils.data import Dataset
import utils


class BioDataset(Dataset):
    # TODO: vedere se transform e target_transform sono utili
    def __init__(self, data, labels, word_to_ix,tag_to_ix,transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._preprocess_samples(data, labels)
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
    
    def _preprocess_samples(self, data, labels): #notebook 3
        res = []
        for sentence, label in zip(data, labels):
            res.append((sentence, label))
        # print(res)
        return(res)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sentence = self.samples[index][0]
        labels = self.samples[index][1]
        res = utils.sentence_to_ix(self.word_to_ix,sentence),utils.label_to_ix(self.tag_to_ix,labels)
        #print(res)
         
        
        
        
        return res
