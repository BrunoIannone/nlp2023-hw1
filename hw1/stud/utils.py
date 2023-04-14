import os
import json
import matplotlib.pyplot as plt
import torch
EMBEDDING_DIM = 32
HIDDEN_DIM = 128
EPOCHS_NUM = 2
LAYERS_NUM = 1
DIRECTORY_NAME = os.path.dirname(__file__)
LEARNING_RATE = 0.001
CHANCES = 5
BATCH_SIZE = 32


def build_vocabulary(sentences):  # {word:idx}
    dict = {}
    idx = 0
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

def plot_logs(logs, title):

    plt.figure(figsize=(8,6))

    plt.plot(list(range(len(logs['train_history']))), logs['train_history'], label='Train loss')
    plt.plot(list(range(len(logs['valid_history']))), logs['valid_history'], label='Test loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.show()

