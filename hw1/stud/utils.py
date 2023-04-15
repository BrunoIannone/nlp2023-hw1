import os
import json
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
EMBEDDING_DIM = 32
HIDDEN_DIM = 128
EPOCHS_NUM = 20
LAYERS_NUM = 1
DIRECTORY_NAME = os.path.dirname(__file__)
LEARNING_RATE = 0.001
CHANCES = 5
BATCH_SIZE = 8192
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_vocabulary(sentences):  # {word:idx}
    dict = {}
    idx = 0
    dict["<PAD>"] = -1
    for sentence in sentences:
        for token in sentence:
            if token not in dict:
                dict[token] = idx
                idx += 1

    return dict


def build_labels(sentences_labels):  # {label:idx}
    dict = {}
    idx = 0
    for labels in sentences_labels:
        for label in labels:
            if label not in dict:
                dict[label] = idx
                idx += 1
    dict["<PAD>"] = idx
    return dict
def build_training_data(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    sentences = []
    labels = []
    while (line):

        json_line = json.loads(line)
        sentences.append(json_line['tokens'])
        labels.append(json_line['labels'])
        line = f.readline()

    f.close()

    return {
        'sentences': sentences,
        'labels': labels

    }
    

def label_to_ix(tag_to_ix,labels):
    res = []
    for label in labels:
        res.append(tag_to_ix[label])
    return res

def sentence_to_ix(word_to_ix,sentence):
    res = []
    for word in sentence:
        if word not in word_to_ix:
            res.append(word_to_ix["UNK"])
        else:
            res.append(word_to_ix[word])
    return res

def plot_logs(logs, title):

    plt.figure(figsize=(8,6))

    plt.plot(list(range(len(logs['train_history']))), logs['train_history'], label='Train loss')
    plt.plot(list(range(len(logs['valid_history']))), logs['valid_history'], label='Test loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.show()

def collate_fn(sentence):
    (xx, yy) = zip(*sentence)
    xx = [torch.tensor(x) for x in xx ]
    yy = [torch.tensor(y) for y in yy ]

    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    #print(xx)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    #print(xx_pad)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    
    return xx_pad.to(DEVICE), yy_pad.to(DEVICE), x_lens, y_lens
    

    

   