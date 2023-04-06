#import train as tr
#import implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import dataloader
#import bioclassifier as bio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sl = dataloader.build_sentence('/home/bruno/Desktop/nlp2023-hw1/data/train.jsonl')
dict = dataloader.build_dictionary(sl['sentences'])

print(dict)
""" EMBEDDING_DIM = 6
HIDDEN_DIM = 6
epochs=100
model = bio.BioClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

trainer = tr.Trainer(model,optimizer,device) 
avg_loss = trainer.train(train_dataset, output_folder, epochs)
 """