import json
import torch
from torch.utils.data import Dataset
import utils
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import List


class BioDataset(Dataset):
    """BIO dataset class
    """
    # TODO: vedere se transform e target_transform sono utili
    def __init__(self, data: List[str], labels: List[str], word_to_ix:dict,tag_to_ix:dict,transform=None, target_transform=None):
        """Constructor for the BIO dataset

        Args:
            data (List[str]): List of sentence tokens
            labels (List[str]): List of sentences labels
            word_to_ix (dict): dictionary with structure {word:index}
            tag_to_ix (dict): dictionary with structure {label:index}
            transform (_type_, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
        """
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

    def __getitem__(self, index:int):
        """Get item for dataloader input

        Args:
            index (int): index to access

        Returns:
            tuple: a tuple containing in tuple[0] a list of token indexes (sentence tokens to their vocabulary indexes) and in tuple[1] their labels converted in indexes
        """
        sentence = self.samples[index][0]
        labels = self.samples[index][1]
        res = utils.sentence_to_ix(self.word_to_ix,sentence), utils.label_to_ix(self.tag_to_ix,labels)
        
        
        
        return res
