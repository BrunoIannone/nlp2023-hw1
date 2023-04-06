import numpy as np

from tqdm.auto import tqdm

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Model


# ○ It should initialize your StudentModel class.

def build_model(device: str) -> Model:

    #Chiamo il costruttore di StudentModel

    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    first_commit = 'hello git'
    return RandomBaseline()


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):
    #● In hw1/stud/implementation.py implement the StudentModel class
#    ○ Load your model and use it in the predict method 
#    ○ You must respect the signature of the predict method! 
#    ○ You can add other methods9 (i.e. the constructor) """

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary


  

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        pass
