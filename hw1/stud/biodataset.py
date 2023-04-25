from torch.utils.data import Dataset
import utils
from typing import List


class BioDataset(Dataset):
    """BIO dataset class
    """
    # TODO: vedere se transform e target_transform sono utili
    def __init__(self, sentences: List[str], labels: List[str], word_to_idx:dict,labels_to_idx:dict,transform=None, target_transform=None):
        """Constructor for the BIO dataset

        Args:
            sentences (List[str]): List of sentence tokens
            labels (List[str]): List of sentences labels
            word_to_idx (dict): dictionary with structure {word:index}
            labels_to_idx (dict): dictionary with structure {label:index}
            transform (_type_, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
        """
        self.sentences = sentences
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._preprocess_samples(sentences, labels)
        self.word_to_idx = word_to_idx
        self.labels_to_idx = labels_to_idx
    
    def _preprocess_samples(self, sentences, labels): #notebook 3
        res = []
        for sentence, label in zip(sentences, labels):
            res.append((sentence, label))
        return res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index:int):
        """Get item for dataloader input

        Args:
            index (int): index to access

        Returns:
            tuple: a tuple containing in tuple[0] a list of token indexes (sentence tokens converted into their vocabulary indexes) and in tuple[1] their labels converted in indexes
        """
        sentence = self.samples[index][0]
        labels = self.samples[index][1]
        res = utils.sentence_to_idx(self.word_to_idx,sentence), utils.label_to_idx(self.labels_to_idx,labels)
        
        
        
        return res
