import train as tr
#import implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
import random
import biodataset
import bioclassifier as bio
import os
import utils


device = utils.DEVICE

training_data = utils.build_training_data(
    os.path.join(utils.DIRECTORY_NAME, '../../data/train.jsonl'))
valid_data = utils.build_training_data(
    os.path.join(utils.DIRECTORY_NAME, '../../data/dev.jsonl'))
test_data = utils.build_training_data(
    os.path.join(utils.DIRECTORY_NAME, '../../data/test.jsonl'))

tag_to_ix = utils.build_labels(training_data["labels"])
# print(tag_to_ix)

word_to_ix = utils.build_vocabulary(training_data["sentences"])


train_dataset = biodataset.BioDataset(
    training_data["sentences"], training_data['labels'], word_to_ix, tag_to_ix)

valid_dataset = biodataset.BioDataset(
    valid_data["sentences"], valid_data['labels'], word_to_ix, tag_to_ix)

test_dataset = biodataset.BioDataset(
    test_data["sentences"], test_data['labels'], word_to_ix, tag_to_ix)


train_dataloader = DataLoader(
    train_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn,shuffle=True)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn,shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn,shuffle=True)


# print(bio_valid_dataset.samples)

# print(valid_dataloader)

# print(round(1.6*pow(len(word_to_ix),1/4)))
# print(dict)
# print(tag_to_ix)
# print(tag_to_ix)
# round(1.6*pow(len(word_to_ix),1/4)
model = bio.BioClassifier(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
                          len(word_to_ix), len(tag_to_ix), utils.LAYERS_NUM, device)
loss_function = nn.CrossEntropyLoss(ignore_index=11)
optimizer = optim.Adam(model.parameters(), lr=utils.LEARNING_RATE)

trainer = tr.Trainer(model, optimizer, device)
#logs = trainer.train(loss_function, train_dataloader, word_to_ix,
             #      tag_to_ix, valid_dataloader, utils.EPOCHS_NUM)
#print(trainer.validation(valid_dataloader, tag_to_ix,
      #word_to_ix, utils.EPOCHS_NUM-1, loss_function))
#utils.plot_logs(logs, 'Train vs Test loss')
print(trainer.test(train_dataloader,306))