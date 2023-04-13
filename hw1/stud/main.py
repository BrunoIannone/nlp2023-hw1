import train as tr
#import implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
import random
import dataloader
import bioclassifier as bio
import os
import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
training_data = utils.build_training_data(
    os.path.join(utils.DIRECTORY_NAME, '../../data/train.jsonl'))

tag_to_ix = utils.build_labels(training_data["labels"])
# print(tag_to_ix)

word_to_ix = utils.build_vocabulary(training_data["sentences"])
bio_dataset = dataloader.BioDataset(
    training_data["sentences"], training_data['labels'])

train_dataloader = DataLoader(bio_dataset, batch_size=1)
valid_data = utils.build_training_data(
    os.path.join(utils.DIRECTORY_NAME, '../../data/dev.jsonl'))
bio_valid_dataset = dataloader.BioDataset(
    valid_data["sentences"], valid_data['labels'])

valid_dataloader = DataLoader(bio_valid_dataset, batch_size=1)

# print(round(1.6*pow(len(word_to_ix),1/4)))
# print(dict)
# print(tag_to_ix)

# round(1.6*pow(len(word_to_ix),1/4)
model = bio.BioClassifier(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
                          len(word_to_ix), len(tag_to_ix), utils.LAYERS_NUM, device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=utils.LEARNING_RATE)

trainer = tr.Trainer(model, optimizer, device)
#logs = trainer.train(loss_function, bio_dataset.samples, word_to_ix,
#                    tag_to_ix, bio_valid_dataset.samples, utils.EPOCHS_NUM)
print(trainer.validation(bio_valid_dataset.samples,tag_to_ix,word_to_ix,utils.EPOCHS_NUM-1,loss_function))
#utils.plot_logs(logs, 'Train vs Test loss')
