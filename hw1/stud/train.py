import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import os 
import utils
import random
import time


class Trainer():
    
    def __init__(self, model, optimizer, device):

        self.device = device

        self.model = model
        self.optimizer = optimizer

        # starts requires_grad for all layers
        self.model.train()  # we are using this model for training (some layers have different behaviours in train and eval mode)
        self.model.to(self.device)  # move model to GPU if available

    def prepare_sequence(self,seq, to_ix):
        idxs  = []
        for w in seq:
            if w in to_ix:
                idxs.append(to_ix[w])
            else:
                idxs.append(to_ix["UNK"])
        
        return torch.tensor(idxs, dtype=torch.long)
    
    def train(
    self,
    
    
    loss_function,
    training_data,
    word_to_ix,
    tag_to_ix,
    bio_valid_dataset,
    epochs: int = 5,
    
    verbose: bool = True
):
        chance = utils.CHANCES
        last_loss = None
        train_log = []
        valid_log = []
        for epoch in tqdm(range(epochs),total = epochs,leave = False, desc = "Epochs"):  
            #print(epoch)
            random.shuffle(training_data)
            losses = []

            for sentence, tags in tqdm(training_data,total = len(training_data), leave = False, desc = "Training"):
                #print(sentence)
                #print(word_to_ix)
                
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                
                sentence_in = self.prepare_sequence(sentence, word_to_ix).to(self.device)
                
                targets = self.prepare_sequence(tags, tag_to_ix).to(self.device)
                #print("SENTENCE" + str(sentence))


                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss= loss_function(tag_scores, targets)
                losses.append(loss)

                loss.backward()
                self.optimizer.step()
                #self.validation(sentence,tag_to_ix,word_to_ix,0,loss_function)
                
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
      
    def validation(self,valid_data,tag_to_ix,word_to_ix,epoch,loss_function,):
        self.model.load_state_dict(torch.load(os.path.join(utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch))))
        self.model.eval()
        tot = len(valid_data)
        loss_avg = 0
        right = 0
        losses = []
        with torch.no_grad():
            for sentence, tag in tqdm(valid_data,total = len(valid_data),leave = False, desc = "Validation"):
                #print(sentence)
                inputs = self.prepare_sequence(sentence, word_to_ix).to(self.device)
                targets = self.prepare_sequence(tag, tag_to_ix).to(self.device)

                prediction = self.model(inputs)
                #print(prediction)
                #tag_ix = utils.label_to_ix(tag_to_ix,tag)
                #  calling optimizer.step()
                loss = loss_function(prediction, targets)
                losses.append(loss)
                #print(loss_avg)
                prediction_list = []

                for row in prediction:
                    
                    prediction_list.append(list(row).index(max(row)))
                #print(prediction_list)
                if prediction_list == list(targets):
                    right+=1
                    
                    """ print("Right" + str(right))
                    print("PREDICTION LIST" + str(prediction_list))
                    print("TAG IX" + str(tag_ix))
                else:
                    print("PREDICTION LIST" + str(prediction_list))
                    print("TAG IX" + str(tag_ix)) """
            
        print(" Precision: " + str((right/tot)*100))
        self.model.train()
        return sum(losses)/len(losses)




            

