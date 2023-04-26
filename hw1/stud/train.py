import torch
import torch.nn as nn
from tqdm.auto import tqdm
import os
import utils
from torch.utils.data import Dataset
from seqeval.metrics import f1_score


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
        chance = utils.CHANCES
        last_loss = None
        train_log = []
        valid_log = []
        max_valid = 0
        for epoch in tqdm(range(epochs),desc="Epochs"):
            self.model.train()

            losses = []
            for _, (sentence, labels, sentence_len, labels_len) in tqdm(enumerate(training_data),desc="Train"):

                self.model.zero_grad()

                predictions = self.model((sentence, sentence_len))
                
                loss = self.loss_function(torch.transpose(predictions, 1, 2), labels)
                losses.append(loss)

                loss.backward()
                self.optimizer.step()

            train_log.append((sum(losses)/len(losses)).item())
            print(" Train Loss: " + str(float(sum(losses)/len(losses))) + "\n")
            valid_loss,f1 = self.validation(valid_dataset)
            print(" Valid loss: " + str(float(valid_loss)) + "\n")
            #if last_loss != None and valid_loss > last_loss:
            #    chance -= 1
            #    print(" LOSS NOT LOWERING => chance = " + str(chance))
            #    if chance <= 0:
            #        break
            #last_loss = valid_loss
            valid_log.append(valid_loss.item())
            if(max_valid<f1):
                max_valid = f1
                max_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(
                utils.DIRECTORY_NAME, 'max.pt'))
        print("Maximum F1 was: " + str(max_valid) + " at epoch: " + str(max_epoch))

        return {
            "train_history": train_log,
            "valid_history": valid_log
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
                transpose_pred = torch.transpose(predictions, 1, 2)
                loss = self.loss_function(transpose_pred, labels)
                losses.append(loss)
                _, predicted_labels = transpose_pred.max(1)
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
                transpose_pred = torch.transpose(predictions, 1, 2)
                _, predicted_labels = transpose_pred.max(1)
                predicted_labels = utils.idx_to_label(
                    self.idx_to_labels, predicted_labels.tolist())
                total_pred.extend(predicted_labels)
                labels = utils.idx_to_label(self.idx_to_labels, labels.tolist())
                total_labels.extend(labels)

        return f1_score(total_labels, total_pred, mode='strict')
