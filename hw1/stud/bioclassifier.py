import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

class BioClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, layers_num, device):
        super(BioClassifier, self).__init__()
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                            bidirectional=False, batch_first=True, dropout=0)

        # The linear layer that maps from hidden state space to tag space
        print(tagset_size)
        self.hidden2tag = nn.Linear(utils.LAYERS_NUM*hidden_dim, tagset_size)

    def forward(self, sentence):
        #print("aio\n")
        #print("NO EMBEDDING: " + str(sentence[0].size()))

        embeds = self.embed(sentence[0])
        #embeds = self.word_embeddings(sentence[0])
        #h0 = torch.rand(self.layers_num*2, embeds.size(0), self.hidden_dim).to(self.device)
        #c0 = torch.rand(self.layers_num*2, embeds.size(0), self.hidden_dim).to(self.device)
        #embeds = nn.utils.rnn.pack_padded_sequence(embeds,len(sentence))
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds,sentence[1], batch_first=True,enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)
        output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        #print("output_padded: " + str(output_padded.size()) + "\n")

        #tag_space = self.hidden2tag(output_padded[:, -1,:])
        tag_space = self.hidden2tag(output_padded)


        
        
        #print("tag_space" + str(tag_space.size()) + "\n")
        tag_scores = F.log_softmax(tag_space,dim=1) ##compute log_softmax along rows
        #print("TAG SCORE:"+ str(tag_scores) + "\n")
        #print("OIA")
        return tag_scores
    def embed(self,sentence):
        return self.word_embeddings(sentence)