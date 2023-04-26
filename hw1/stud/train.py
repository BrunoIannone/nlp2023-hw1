import torch
import torch.nn as nn
from tqdm.auto import tqdm
import os
import utils
from torch.utils.data import Dataset
from seqeval.metrics import f1_score

import time
class Trainer():
    """Class for model training
    """

    def __init__(self, model: nn.Module, optimizer, device: str, loss_function,idx_to_labels:dict):
        """Trainer init class

        Args:
            model (nn.Module): Chosen model to train
            optimizer: Chosen optimizer to use for optimization
            device (str): Chosen device for training
            idx_to_labels (dict): dictionary with structure {index:label}

        """
        
        self.idx_to_labels = idx_to_labels
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model.to(self.device)

    def train(self, training_data: Dataset, valid_dataset: Dataset, epochs: int):
        """Training function

        Args:
            loss_function: Chosen loss function
            training_data (Dataset): Training data
            valid_dataset (Dataset): Validation data
            epochs (int): Number of epochs


        Returns:
            dict: {"train_history" : train_loss_log, "valid_history": valid_loss_log}
        """
        max_epoch = 0
        if(utils.EARLY_STOP):
            chance = utils.CHANCES
            last_f1 = None
        train_log = []
        valid_log = []
        f1_log= []
        max_f1 = 0
        for epoch in tqdm(range(epochs),desc="Epochs"):
            self.model.train()

            losses = []
            for _, (sentence, labels, sentence_len, labels_len) in tqdm(enumerate(training_data),desc="Train"):

                self.model.zero_grad()

                predictions = self.model((sentence, sentence_len))
                
                predictions = predictions.view(-1, predictions.shape[-1])

                labels = labels.view(-1)

                loss = self.loss_function(predictions, labels)
                

                losses.append(loss)

                loss.backward()
                self.optimizer.step()

            train_log.append((sum(losses)/len(losses)).item())
            print(" Train Loss: " + str(float(sum(losses)/len(losses))) + "\n")
            valid_loss,f1 = self.validation(valid_dataset)
            print(" Valid loss: " + str(float(valid_loss)) + "\n")
            if utils.EARLY_STOP and last_f1 != None and f1 > last_f1:
                chance -= 1
                print(" F1 LOWERING => chance = " + str(chance))
                if chance <= 0:
                    break
            if utils.EARLY_STOP:
                last_f1 = f1
            valid_log.append(valid_loss.item())
            f1_log.append(f1.item())
            if(max_f1<f1):
                max_f1 = f1
                max_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(
                utils.DIRECTORY_NAME, 'max.pt'))
                if utils.EARLY_STOP:
                    print("New max F1 reached, restoring chances")
                    chance = utils.CHANCES
        print("Maximum F1 was: " + str(max_f1) + " at epoch: " + str(max_epoch))

        return {
            "train_history": train_log,
            "valid_history": valid_log,
            "f1_history": f1_log
        }

    def validation(self, valid_data: Dataset):
        """Function for model evaluation

        Args:
            valid_data (Dataset): validation data

        Returns:
            tuple: (validation loss (average), F1 score )
        """
        self.model.eval()

        total_pred = []
        total_labels = []
        losses = []
        with torch.no_grad():

            for _, (sentence, labels, sentence_len, labels_len) in tqdm(enumerate(valid_data),desc="Validation"):

                predictions = self.model((sentence, sentence_len))
                
                predictions_view = predictions.view(-1, predictions.shape[-1])

                labels_view = labels.view(-1)

                loss = self.loss_function(predictions_view, labels_view)
                losses.append(loss)

                predicted_labels = torch.argmax(predictions,-1)
                predicted_labels = utils.idx_to_label(
                    self.idx_to_labels, predicted_labels.tolist())
                total_pred.extend(predicted_labels)
                labels = utils.idx_to_label(self.idx_to_labels, labels.tolist())
                total_labels.extend(labels)

        f1 = f1_score(total_labels, total_pred, mode='strict')
        print("F1: " + str(f1) +  "\n")
        print("\n")

        return sum(losses)/len(losses),f1

    def test(self, test_data: Dataset):
        """Function for model testing

        Args:
            test_data (Dataset): Test data
            epoch (int): epoch number
           

        Returns:
            float: F1 score
        """
        total_pred = []
        total_labels = []

        
        self.model.load_state_dict(torch.load(os.path.join(
            utils.DIRECTORY_NAME, 'max.pt')))
        
        self.model.eval()
        with torch.no_grad():

            for _, (sentence, labels, sentence_len, labels_len) in tqdm(enumerate(test_data), desc="Testing"):
               predictions = self.model((sentence, sentence_len))
                
               predicted_labels = torch.argmax(predictions,-1)

               predicted_labels = utils.idx_to_label(
                   self.idx_to_labels, predicted_labels.tolist())
               total_pred.extend(predicted_labels)
               labels = utils.idx_to_label(self.idx_to_labels, labels.tolist())
               total_labels.extend(labels)

        return f1_score(total_labels, total_pred, mode='strict')
