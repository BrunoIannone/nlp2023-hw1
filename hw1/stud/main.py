import train as tr
#import implementation
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
import biodataset
import bioclassifier as bio
import os
import utils
import torch


device = utils.DEVICE

training_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/train.jsonl'))
valid_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/dev.jsonl'))
test_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/test.jsonl'))

tag_to_ix = utils.build_labels_vocabulary(training_data["labels"])

word_to_ix = utils.build_tokens_vocabulary(training_data["sentences"])

print(len(word_to_ix))

train_dataset = biodataset.BioDataset(
    training_data["sentences"], training_data['labels'], word_to_ix, tag_to_ix)

valid_dataset = biodataset.BioDataset(
    valid_data["sentences"], valid_data['labels'], word_to_ix, tag_to_ix)

test_dataset = biodataset.BioDataset(
    test_data["sentences"], test_data['labels'], word_to_ix, tag_to_ix)


train_dataloader = DataLoader(
    train_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn, shuffle=True)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn, shuffle=True)


model = bio.BioClassifier(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
                          len(word_to_ix), len(tag_to_ix), utils.LAYERS_NUM, device)
loss_function = nn.CrossEntropyLoss(ignore_index=len(tag_to_ix)-1)
optimizer = optim.Adam(model.parameters(), lr=utils.LEARNING_RATE)


trainer = tr.Trainer(model, optimizer, device, loss_function, word_to_ix,
                     tag_to_ix)
logs = trainer.train(train_dataloader, valid_dataloader, utils.EPOCHS_NUM)
utils.plot_logs(logs, 'Train vs Test loss')
results = trainer.test(test_dataloader, utils.EPOCHS_NUM-1)
#TODO: automatizzare il carico dei pesi
print(results)
