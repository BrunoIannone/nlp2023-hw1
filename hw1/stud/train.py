import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import os
import utils
import random
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from seqeval.metrics import f1_score


class Trainer():
    """Class for model training
    """

    def __init__(self, model: nn.Module, optimizer, device: str, loss_function, word_to_ix: dict, tag_to_ix: dict):
        """_summary_

        Args:
            model (nn.Module): Chosen model to train
            optimizer: Chosen optimizer to use for optimization
            device (str): Chosen device for training
            word_to_ix (dict): (Vocabulary) dictionary with structure {token:index}
            tag_to_ix (dict):  dictionary with structure {label:index}
        """
        self.tag_to_ix = tag_to_ix
        self.word_to_ix = word_to_ix
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model.to(self.device)

    def train(self, training_data: Dataset, valid_dataset: Dataset, epochs: int):
        """Training function

        Args:
            loss_function: Chosen loss function
            training_data (Dataset): Training data from Data Loader

            valid_dataset (Dataset): BIO dataset class
            epochs (int): Number of epochs


        Returns:
            dict: dictionary wit train/valid losses {"train_history" : train_loss_log, "valid_history": valid_loss_log}
        """

        chance = utils.CHANCES
        last_loss = None
        train_log = []
        valid_log = []
        for epoch in tqdm(range(epochs), total=epochs, leave=False, desc="Epochs"):
            # print(epoch)
            self.model.train()

            losses = []
            # ,total = len(training_data), leave = True):
            for _, (sentence, tags, sentence_len, tags_len) in enumerate(tqdm(training_data, leave=False)):

                self.model.zero_grad()

                tag_scores = self.model((sentence, sentence_len))

                loss = self.loss_function(torch.transpose(tag_scores, 1, 2), tags)
                losses.append(loss)

                loss.backward()
                self.optimizer.step()

            train_log.append((sum(losses)/len(losses)).item())
            print(" Train Loss: " + str(sum(losses)/len(losses)))
            valid_loss = self.validation(valid_dataset)
            print(" Valid loss: " + str(float(valid_loss)))
            if last_loss != None and valid_loss > last_loss:
                chance -= 1
                print(" LOSS NOT LOWERING => chance = " + str(chance))
                if chance <= 0:
                    break
            last_loss = valid_loss
            valid_log.append(valid_loss.item())
        torch.save(self.model.state_dict(), os.path.join(
            utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch)))

        return {
            "train_history": train_log,
            "valid_history": valid_log
        }

    def validation(self, valid_data: Dataset):
        """Function for model evaluation

        Args:
            valid_data (Dataset): BIO dataset class

        Returns:
            float: average valid loss
        """
        self.model.eval()

        total_pred = []
        total_tags = []
        losses = []
        with torch.no_grad():

            # ,total = len(valid_data), leave = True, desc = "Validation"):
            for _, (sentence, tags, sentence_len, tags_len) in enumerate(tqdm(valid_data, leave=False)):

                prediction = self.model((sentence, sentence_len))
                transpose_pred = torch.transpose(prediction, 1, 2)
                loss = self.loss_function(transpose_pred, tags)
                losses.append(loss)
                _, predictions = transpose_pred.max(1)

                predictions = utils.ix_to_label(
                    self.tag_to_ix, predictions.tolist())
                total_pred.extend(predictions)
                tags = utils.ix_to_label(self.tag_to_ix, tags.tolist())
                total_tags.extend(tags)
        print(f1_score(total_tags, total_pred, mode='strict'))

        return sum(losses)/len(losses)

    def test(self, test_data: Dataset, epoch: int):
        """Function for model testing

        Args:
            test_data (Dataset): Test data
            epoch (int): epoch number
            tag_to_ix (dict):  dictionary with structure {label:index}

        Returns:
            float: F1 score
        """
        total_pred = []
        total_tags = []

        right = 0
        self.model.load_state_dict(torch.load(os.path.join(
            utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch))))
        tot = 0
        self.model.eval()
        with torch.no_grad():

            for _, (sentence, tags, sentence_len, tags_len) in tqdm(enumerate(test_data), total=len(test_data), leave=True, desc="Testing", position=100):
                prediction = self.model((sentence, sentence_len))
                _, predictions = torch.transpose(prediction, 1, 2).max(1)
                predictions = utils.ix_to_label(
                    self.tag_to_ix, predictions.tolist())
                total_pred.extend(predictions)

                # print(predictions)
                tags = utils.ix_to_label(self.tag_to_ix, tags.tolist())
                total_tags.extend(tags)

        return f1_score(total_tags, total_pred, mode='strict')
