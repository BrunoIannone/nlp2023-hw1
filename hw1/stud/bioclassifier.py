import torch
import torch.nn as nn
import utils as utils


class BioClassifier(nn.Module):
    """BIO classifier class
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, labelset_size: int, layers_num: int, device: str, embedding):
        """Init class for the BIO classifier

        Args:
            embedding_dim (int): Embedding dimension
            hidden_dim (int): Hidden dimension
            vocab_size (int): Vocabulary size
            labelset_size (int): Number of classes
            layers_num (int): Number of layers of the LSTM
            device (str): Device for calculation
        """
        super(BioClassifier, self).__init__()
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.device = device

        if(embedding):  # note that the vocabulary must have an entry for padding and unk
            self.word_embeddings = nn.Embedding.from_pretrained(
                embedding, freeze=True)
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                            bidirectional=utils.BIDIRECTIONAL, batch_first=True, dropout=utils.DROPOUT_LSTM)

        if(utils.DROPOUT_LAYER > 0):
            self.dropout_layer = nn.Dropout(utils.DROPOUT_LAYER)

        if(utils.DROPOUT_EMBED > 0):
            self.dropout_embed = nn.Dropout(utils.DROPOUT_EMBED)

        if(utils.BIDIRECTIONAL):

            self.hidden2labels = nn.Linear(2*hidden_dim, labelset_size)
        else:
            self.hidden2labels = nn.Linear(hidden_dim, labelset_size)

    def forward(self, sentence: tuple):
        """Model forward pass

        Args:
        sentence (tuple): (Tensor[padded_sentences], List[lenghts (int)]) N.B.: lenghts refers to the original non padded sentences

        Returns:
            Tensor: Model predictions
        """
        embeds = self.word_embeddings(sentence[0])
        if(utils.DROPOUT_EMBED > 0):
            embeds = self.dropout_embed(embeds)

        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, sentence[1], batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(embeds)

        output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)

        if(utils.DROPOUT_LAYER > 0):
            output_padded = self.dropout_layer(output_padded)

        labels_space = self.hidden2labels(output_padded)
        return labels_space
