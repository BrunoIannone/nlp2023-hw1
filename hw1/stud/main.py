import train as tr
#import implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataloader
import bioclassifier as bio
import os

dirname = os.path.dirname(__file__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
training_data = dataloader.build_training_data(
    os.path.join(dirname, '../../data/train.jsonl'))
tag_to_ix = dataloader.build_labels(training_data["labels"])

word_to_ix = dataloader.build_vocabulary(training_data["sentences"])

# print(dict)
# print(tag_to_ix)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
epochs = 100
model = bio.BioClassifier(EMBEDDING_DIM, HIDDEN_DIM,
                          len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

trainer = tr.Trainer(model, optimizer, device)
trainer.train(loss_function, training_data, word_to_ix, tag_to_ix, 5)
#avg_loss = trainer.train(train_dataset, output_folder, epochs)
