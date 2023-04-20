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

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                            bidirectional=utils.BIDIRECTIONAL, batch_first=True, dropout=0)
        if(utils.BIDIRECTIONAL):

            self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        

        embeds = self.embed(sentence[0])
        
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds,sentence[1], batch_first=True,enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)
        
        output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.hidden2tag(output_padded)

        return tag_space
    def embed(self,sentence):
        return self.word_embeddings(sentence)