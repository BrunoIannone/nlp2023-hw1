import os
import json
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence

###HYPERPARAMETERS###
EMBEDDING_DIM = 32
LAYERS_NUM = 2
HIDDEN_DIM = 128 
EPOCHS_NUM = 100
BIDIRECTIONAL = True
DIRECTORY_NAME = os.path.dirname(__file__)
LEARNING_RATE = 0.001
CHANCES = 5
BATCH_SIZE = 1#4096 #2^12
#####################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_tokens_vocabulary(sentences:list[list[str]],padding:bool = False):  # {word:idx}
    """Create a vocabulary from tokens in sentences with padding having index -1

    Args:
        sentences (list[list[str]]): list of list of strings
        padding (bool): if TRUE, add padding <PAD> key with  value -1. Default = False

    Returns:
        dictionary: Returns a dictionary with the structure {token:index}, if padding=True: {token:index,...,<PAD>:-1}  
    """
    dict = {}
    idx = 0
    if padding:
        dict["<PAD>"] = -1
    for sentence in sentences:
        for token in sentence:
            if token not in dict:
                dict[token] = idx
                idx += 1

    return dict


def build_labels_vocabulary(sentences_labels: list[list[str]],padding:bool = False):  # {label:idx}
    """Converts labels in integral indexes with padding having the greater index

    Args:
        sentences_labels (list[list[str]]): List of list of strings of labels
        padding (bool): if TRUE, add padding <PAD> key with the greatest value. Default = False

    Returns:
        dictionary: Returns a dictionary with the structure {label:index}
    """
    dict = {}
    idx = 0
    for labels in sentences_labels:
        for label in labels:
            if label not in dict:
                dict[label] = idx
                idx += 1
    if padding:
        dict["<PAD>"] = idx
    return dict

def build_data_from_jsonl(file_path:str): 
    """Split the JSONL file in file_path in sentences and relative labels 

    Args:
        file_path (string): path to JSONL file

    Returns:
        dictionary: return a dictionary with keys "sentences" and "labels" with keys list of list of strings: {sentences: [sentences], labels:[labels]}
    """
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in "+ str(file_path))
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
    

def label_to_ix(tag_to_ix:dict,labels:list(str)):
    """Converts labels string in integer indexes. 
       

    Args:
        tag_to_ix (dictionary): dictionary with structure {token:index} 
        labels (list(string)): List of labels (stings)

    Returns:
        list: list of integers that represent labels indexes
    """
    res = []
    for label in labels:
        res.append(tag_to_ix[label])
    return res

def sentence_to_ix(word_to_ix:dict,sentence:list(str)):
    """Converts tokens of strings in their indexes. If a token is unknown, it's index is the <UNK> key value

    Args:
        word_to_ix (dict): dictionary with structure {word:index}
        sentence (list): list of tokens (strings)

    Returns:
        list: list of integers that represent tokens indexes
    """
    res = []
    for word in sentence:
        if word not in word_to_ix:
            res.append(word_to_ix["UNK"])
        else:
            res.append(word_to_ix[word])
    return res
def ix_to_label(tag_to_ix, src_label):
    out_label = []
    temp = []
    for label_list in src_label:
        #print(label_list)
        #time.sleep(5)
        temp = []
        for label in label_list:
            #print(label)
            #time.sleep(5)
            for key in tag_to_ix:
                #print(key)
               
                if tag_to_ix[key] == label:
                    if(key == "<PAD>"):
                        temp.append("O")
                    else:
                        temp.append(key)
                    
        

        out_label.append(temp)

    #print(out_label)              
    return out_label


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
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    
    return xx_pad.to(DEVICE), yy_pad.to(DEVICE), x_lens, y_lens
    

    

   