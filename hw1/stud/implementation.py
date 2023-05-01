import numpy as np

from tqdm.auto import tqdm

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import stud.bioclassifier_implementation as bioclassifier
import stud.utils as utils
from model import Model


IMPL_DIRECTORY_NAME = os.path.dirname(__file__)


def build_model(device: str) -> Model:

    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates

    return StudentModel(device)


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
            [str(np.random.choice(self._options, 1, p=self._weights)[0])
             for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device):
        self.vocab = self.load_vocabularies(IMPL_DIRECTORY_NAME)
        self.model = bioclassifier.BioClassifier(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
                                                 len(self.vocab["word_to_idx"]), len(self.vocab["labels_to_idx"]), utils.LAYERS_NUM, device, None)
        self.model.load_state_dict(torch.load(os.path.join(
            IMPL_DIRECTORY_NAME, '../../model/0.702--0.721.pt'), map_location=torch.device(device)))

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.eval()
        with torch.no_grad():
            res = []
            for sentence in tokens:

                sentence_len = len(sentence)
                sentence = utils.word_to_idx(
                    self.vocab["word_to_idx"], sentence)
                predictions = self.model(
                    (torch.tensor(sentence).view((1, sentence_len)), [sentence_len]))
                predicted_labels = torch.argmax(predictions, -1)
                predicted_labels = utils.idx_to_label(
                    self.vocab["idx_to_labels"], predicted_labels.tolist())

                # appending would violate the signature
                res.extend(predicted_labels)
                # as idx_to_label returns List[List[str]]

            return res

    def load_vocabularies(self, path: str):
        """_summary_

        Args:
            path (str): path to vocabularies word_to_idx and viceversa, labels_to_idx and viceversa

        Returns:
            dict: a dictionary containing the four others dictionaries: {"word_to_idx":word_to_idx,"idx_to_word":idx_to_word,"labels_to_idx":labels_to_idx,"idx_to_labels":idx_to_labels}
        """
        vocab = {}
        with open(os.path.join(path, "../../model/word_to_idx.txt"), "r") as fp:
            vocab["word_to_idx"] = json.load(fp)
            fp.close()

        with open(os.path.join(path, "../../model/idx_to_word.txt"), "r") as fp:
            vocab["idx_to_word"] = json.load(fp, object_hook=utils.str_to_int)
            fp.close()

        with open(os.path.join(path, "../../model/labels_to_idx.txt"), "r") as fp:
            vocab["labels_to_idx"] = json.load(fp)
            fp.close()

        with open(os.path.join(path,  "../../model/idx_to_labels.txt"), "r") as fp:
            vocab["idx_to_labels"] = json.load(
                fp, object_hook=utils.str_to_int)
            fp.close()
        return vocab
