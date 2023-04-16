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
        self.model.train()
        for epoch in tqdm(range(epochs),total = epochs,leave = False, desc = "Epochs"):  
            #print(epoch)
            losses = []
            for _,(sentence,tags,sentence_len,tags_len) in tqdm(enumerate(training_data),total = len(training_data), leave = True):
                #print(sentence)

                #embeds = self.model.embed(sentence)
                #print(word_to_ix)
               
                
                #tags = pack_padded_sequence(tags)
                
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                
                #sentence_in = self.prepare_sequence(sentence, word_to_ix).to(self.device)
                
                #targets = self.prepare_sequence(tags, tag_to_ix).to(self.device)
                #print("SENTENCE" + str(sentence))


                # Step 3. Run our forward pass.
                tag_scores= self.model((sentence,sentence_len))
                #print("SCORES SIZE:" +str(torch.transpose(tag_scores,1,2).size()) + "\n")

                #print("TAGS SIZE:" +str(tags.size()) + "\n")
                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()

                loss= loss_function(torch.transpose(tag_scores,1,2), tags)
                losses.append(loss)

                loss.backward()
                self.optimizer.step()
                self.validation(bio_valid_dataset,tag_to_ix,word_to_ix,0,loss_function)
                
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
      
    def validation(self,valid_data,tag_to_ix,word_to_ix,epoch,loss_function):
        #self.model.load_state_dict(torch.load(os.path.join(utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch))))
        self.model.eval()
        tot = len(valid_data)
        print(tot)
        loss_avg = 0
        right = 0
        losses = []
        with torch.no_grad():
            
            for _,(sentence,tags,sentence_len,tags_len) in tqdm(enumerate(valid_data),total = len(valid_data), leave = True, desc = "Validation"):
                
                #print(sentence)
                #inputs = self.prepare_sequence(sentence, word_to_ix).to(self.device)
                #targets = self.prepare_sequence(tag, tag_to_ix).to(self.device)
                
                prediction = self.model((sentence,sentence_len))
                #print(prediction)
                #tag_ix = utils.label_to_ix(tag_to_ix,tag)
                #  calling optimizer.step()
                #print(prediction)
                loss = loss_function(torch.transpose(prediction,1,2), tags)
                losses.append(loss)
                #print(loss_avg)
                #prediction_list = []

                #for row in prediction:
                    
                #    prediction_list.append(list(row).index(max(row)))
                #print(prediction_list)
                #if prediction_list == list(targets):
                   # right+=1
                    
                #    #print("Right" + str(right))
                    #print("PREDICTION LIST" + str(prediction_list))
                    #print("TAG IX" + str(tag_ix))
                #else:
                   # print("PREDICTION LIST" + str(prediction_list))
                    #print("TAG IX" + str(tag_ix)) 
                #print(prediction)
                #print(prediction)
                #print(tags.size())
               # _,predictions = prediction.max(1)
                #print(predictions)


                #print("PREDICTION " + str(predictions))
                #print("TARGETS " + str(targets))
                #print()
                
                #right += (torch.transpose(predictions,1,2) == tags).sum()
                #print("RIGHT", str(right))
                #tot += predictions.size(0) 
            
        #print(" Precision: " + str((right/tot)*100))
        self.model.train()
        return sum(losses)/len(losses)

    def test (self, test_data,epoch):
        right = 0
        model = self.model.load_state_dict(torch.load(os.path.join(utils.DIRECTORY_NAME, 'state_{}.pt'.format(epoch))))
        tot  = 0
        self.model.eval()
        #embeddings = self.model.word_embeddings.weight
        with torch.no_grad():
            for _,(sentence,tags,sentence_len,tags_len) in tqdm(enumerate(test_data),total = len(test_data), leave = True, desc = "Testing",position=100):
                prediction = self.model((sentence,sentence_len))
                print(prediction.size())
                
                _,predictions = torch.transpose(prediction,1,2).max(1)
                tot += sentence.size(0)

                unpadded_tags = []
                #print(predictions)
                #time.sleep(5)
                #predictions = prediction.tolist()
                for i in range(len(tags_len)):
                    row =tags[i].tolist()
                    #print(len(row[:tags_len[i]]))
                    #print(tags_len[i])
                    #print("\n")
                    #unpadded_tags.append(row[:tags_len[i]])
                    
                    #print(predictions[i][:sentence_len[i]].tolist())
                    #print(row[:tags_len[i]])
                    #print("\n")
                    
                    
                
                    if(predictions[i][:sentence_len[i]].tolist() == row[:tags_len[i]]):
                        right +=1 
                        #print(right)
                    #time.sleep(5)
                #if predictions.tolist() == unpadded_tags:
                #    right+=1
                #right += (predictions.tolist() == unpadded_tags).sum()
                
        print(" Precision: " + str((right/tot)*100))







            

