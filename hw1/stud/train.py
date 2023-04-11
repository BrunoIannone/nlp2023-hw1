import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import os 
import utils
import random


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

        chance = 3
        last_loss = -999
        train_log = []
        valid_log = []
        for epoch in tqdm(range(epochs),total = epochs,leave = False, desc = "Epochs"):  # again, normally you would NOT do 300 epochs, it is toy data
            #print(epoch)
            random.shuffle(training_data)
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
                loss = loss_function(tag_scores, targets)
                
                loss.backward()
                self.optimizer.step()
                
                
            print(" Train Loss: "+ str(float(loss)))
            train_log.append(float(loss))
            loss = self.validation(bio_valid_dataset,tag_to_ix,word_to_ix,epoch,loss_function)
            print(" Valid avg loss: " + str(float(loss)))
            if loss > last_loss and last_loss!= -999:
                  chance -=1
                  print(" LOSS NOT LOWERING => chance = " + str(chance))
                  if chance<=0:
                    break
            last_loss = loss
            valid_log.append(float(loss))
        torch.save('.','state_{}.pt'.format(epoch))
        
        return {
            "train_history" : train_log,
            "valid_history": valid_log
        }
        torch.save(self.model.state_dict(),
                       os.path.join(utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch)))
        """ # See what the scores are after training
        with torch.no_grad():
            inputs = self.prepare_sequence(training_data["sentences"][0], word_to_ix).to(self.device)
            tag_scores = self.model(inputs)

            # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # for word i. The predicted tag is the maximum scoring tag.
            # Here, we can see the predicted sequence below is 0 1 2 0 1
            # since 0 is index of the maximum value of row 1,
            # 1 is the index of maximum value of row 2, etc.
            # Which is DET NOUN VERB DET NOUN, the correct sequence!
            print(tag_scores)  """
        
        #print(self.validation(self.model,valid_data,tag_to_ix,word_to_ix))
    
    def validation(self,valid_data,tag_to_ix,word_to_ix,epoch,loss_function):
        #self.model.load_state_dict(torch.load(os.path.join(utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch))))

        tot = len(valid_data)
        loss_avg = 0
        right = 0
        for sentence, tag in tqdm(valid_data,total = len(valid_data),leave = False, desc = "Validation"):
            #print(sentence)
            inputs = self.prepare_sequence(sentence, word_to_ix).to(self.device)
            targets = self.prepare_sequence(tag, tag_to_ix).to(self.device)

            prediction = self.model(inputs)
            tag_ix = utils.label_to_ix(tag_to_ix,tag)
            #  calling optimizer.step()
            loss = loss_function(prediction, targets)
            loss_avg += loss
            #print(loss_avg)
            prediction_list = []
            for row in prediction:
                
                prediction_list.append(list(row).index(max(row)))
            if prediction_list == tag_ix:
                right+=1
                #print("Right" + str(right))
                #print("PREDICTION LIST" + str(prediction_list))
                #print("TAG IX" + str(tag_ix))

        print(" Precision: " + str((right/tot)*100))
        return loss_avg/tot




            

