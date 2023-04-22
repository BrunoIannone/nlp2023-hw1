import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import os 
import utils
import random
import time
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.utils.data import Dataset
from seqeval.metrics import f1_score

class Trainer():
    """Class for model training
    """
    def __init__(self, model:nn.Module, optimizer, device:str):
        """_summary_

        Args:
            model (nn.Module): Chosen model to train
            optimizer: Chosen optimizer to use for optimization
            device (str): Chosen device for training
        """


        self.device = device

        self.model = model
        self.optimizer = optimizer

        # starts requires_grad for all layers
        self.model.train()  # we are using this model for training (some layers have different behaviours in train and eval mode)
        self.model.to(self.device)  # move model to GPU if available

    
    def train(
    
    self,
    
    
    loss_function,
    training_data:Dataset,
    word_to_ix:dict,
    tag_to_ix:dict,
    bio_valid_dataset: Dataset,
    epochs: int,
    ):
        """Training function

        Args:
            loss_function: Chosen loss function
            training_data (Dataset): Training data from Data Loader
            word_to_ix (dict): (Vocabulary) dictionary with structure {token:index}
            tag_to_ix (dict):  dictionary with structure {label:index}
            bio_valid_dataset (Dataset): BIO dataset class
            epochs (int): Number of epochs
            

        Returns:
            dict: dictionary wit train/valid losses {"train_history" : train_loss_log, "valid_history": valid_loss_log}
        """
        chance = utils.CHANCES
        last_loss = None
        train_log = []
        valid_log = []
        self.model.train()
        for epoch in tqdm(range(epochs),total = epochs,leave = False, desc = "Epochs"):  
            #print(epoch)
            losses = []
            for _,(sentence,tags,sentence_len,tags_len) in enumerate(tqdm(training_data,leave=False)): #,total = len(training_data), leave = True):
                
                self.model.zero_grad()

                tag_scores= self.model((sentence,sentence_len))


                loss= loss_function(torch.transpose(tag_scores,1,2), tags)
                losses.append(loss)

                loss.backward()
                self.optimizer.step()
                #self.validation(bio_valid_dataset,tag_to_ix,word_to_ix,0,loss_function)
                
            train_log.append((sum(losses)/len(losses)).item())
            print(" Train Loss: "+ str(sum(losses)/len(losses)))
            valid_loss = self.validation(bio_valid_dataset,tag_to_ix,word_to_ix,epoch,loss_function)
            print(" Valid loss: " + str(float(valid_loss)))
            if last_loss!= None and valid_loss > last_loss  :
                  chance -=1
                  print(" LOSS NOT LOWERING => chance = " + str(chance))
                  if chance<=0:
                    break
            last_loss = valid_loss
            valid_log.append(valid_loss.item())
        torch.save(self.model.state_dict(),os.path.join(utils.DIRECTORY_NAME,'state_{}.pt'.format(epoch)))
        
        return {
            "train_history" : train_log,
            "valid_history": valid_log
        }
      
    def validation(self,valid_data:Dataset,tag_to_ix:dict,word_to_ix,epoch,loss_function):
        
        """Function for model evaluation

        Args:
            valid_data (Dataset): BIO dataset class
            tag_to_ix (dict):  dictionary with structure {label:index}
            loss_function: Chosen loss function

        Returns:
            float: average valid loss
        """
        self.model.eval()
        total_pred=[]
        total_tags = []
        tot = len(valid_data)
        print(tot)
        loss_avg = 0
        right = 0
        losses = []
        with torch.no_grad():
            
            for _,(sentence,tags,sentence_len,tags_len) in enumerate(tqdm(valid_data,leave=False)):#,total = len(valid_data), leave = True, desc = "Validation"):
                               
                prediction = self.model((sentence,sentence_len))
                transpose_pred = torch.transpose(prediction,1,2)
                loss = loss_function(transpose_pred, tags)
                losses.append(loss)
                _,predictions =transpose_pred.max(1)

                predictions = utils.ix_to_label(tag_to_ix,predictions.tolist())
                total_pred.extend(predictions)
                #print(predictions)
                tags = utils.ix_to_label(tag_to_ix,tags.tolist())
                total_tags.extend(tags)
            print(f1_score(total_tags, total_pred,mode = 'strict'))
        self.model.train()

        return sum(losses)/len(losses)

    def test (self, test_data:Dataset,epoch:int,tag_to_ix:dict):
        """Function for model testing

        Args:
            test_data (Dataset): Test data
            epoch (int): epoch number
            tag_to_ix (dict):  dictionary with structure {label:index}

        Returns:
            float: F1 score
        """
        
        
        right = 0
        model = self.model.load_state_dict(torch.load(os.path.join(utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch))))
        tot  = 0
        self.model.eval()
        # TODO: CALCOLARE F1 GLOBALE IN MODO CORRETTO
        with torch.no_grad():
            
            for _,(sentence,tags,sentence_len,tags_len) in tqdm(enumerate(test_data),total = len(test_data), leave = True, desc = "Testing",position=100):
                prediction = self.model((sentence,sentence_len))
                #print(prediction.size())
                
                _,predictions = torch.transpose(prediction,1,2).max(1)
                predictions = utils.ix_to_label(tag_to_ix,predictions.tolist())
                #print(predictions)
                tags = utils.ix_to_label(tag_to_ix,tags.tolist())

                return f1_score(tags, predictions,mode = 'strict')

               
                







            

