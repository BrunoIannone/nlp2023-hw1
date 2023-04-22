import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from typing import List


class BioClassifier(nn.Module):
    """BIO classifier class
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, layers_num: int, device: str):
        """Init class for the BIO classifier

        Args:
            embedding_dim (int): Embedding dimension
            hidden_dim (int): Hidden dimension
            vocab_size (int): Vocabulary size
            tagset_size (int): Number of classes
            layers_num (int): Number of layers of the LSTM
            device (str): Device for calculation
        """
        super(BioClassifier, self).__init__()
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.device = device
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                            bidirectional=utils.BIDIRECTIONAL, batch_first=True, dropout=0)
        if(utils.BIDIRECTIONAL):

            self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: tuple):
        """Model forward pass

        Args:
        sentence (tuple): A tuple containing in sentence[0] a tensor of padded sentences and in sentence[1] a list of original (non padded) lenghts for each sentence. The bird view is (tensor_padded_sentences, list_of_lenghts) 

        Returns:
            Tensor: Model predictions
        """

        embeds = self.embed(sentence[0])

        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, sentence[1], batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)

        output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)
        tag_space = self.hidden2tag(output_padded)

        return tag_space

    def embed(self, sentence:torch.Tensor):
        """Aux function for embedding

        Args:
            sentence (torch.Tensor): padded Tensor of sentences

        Returns:
            Tensor: word embedding fot input tensor of sentences
        """
        
        return self.word_embeddings(sentence)
