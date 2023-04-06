import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import os 



class Trainer():
    
    def __init__(self, model, optimizer, device):

        self.device = device

        self.model = model
        self.optimizer = optimizer

        # starts requires_grad for all layers
        self.model.train()  # we are using this model for training (some layers have different behaviours in train and eval mode)
        self.model.to(self.device)  # move model to GPU if available

    def prepare_sequence(self,seq, to_ix):
        #seq = seq[0]
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)
    
    def train(
    self,
    
    
    loss_function,
    training_data,
    word_to_ix,
    tag_to_ix,
    epochs: int = 5,
    
    verbose: bool = True
):

        
        
        for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
            print(epoch)
            for sentence, tags in zip(training_data["sentences"],training_data["labels"]):
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

        # See what the scores are after training
        with torch.no_grad():
            inputs = self.prepare_sequence(training_data["sentences"][0], word_to_ix).to(self.device)
            tag_scores = self.model(inputs)

            # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # for word i. The predicted tag is the maximum scoring tag.
            # Here, we can see the predicted sequence below is 0 1 2 0 1
            # since 0 is index of the maximum value of row 1,
            # 1 is the index of maximum value of row 2, etc.
            # Which is DET NOUN VERB DET NOUN, the correct sequence!
            print(tag_scores) 