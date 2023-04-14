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
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                            bidirectional=False, batch_first=True, dropout=0)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(utils.LAYERS_NUM*hidden_dim, tagset_size)

    def forward(self, sentence):
        

        embeds = self.word_embeddings(sentence)
        embeds = nn.utils.rnn.pack_padded_sequence(embeds,len(sentence))
        #h0 = torch.rand(self.layers_num*2, embeds.size(0), self.hidden_dim).to(self.device)
        #c0 = torch.rand(self.layers_num*2, embeds.size(0), self.hidden_dim).to(self.device)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        tag_space = self.hidden2tag(lstm_out[:, -1,:])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
