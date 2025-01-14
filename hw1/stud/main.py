import train as tr
import torch.nn as nn
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
import biodataset
import bioclassifier as bio
import os
import utils
import vocabulary
import torch

device = utils.DEVICE

training_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/train.jsonl'))
valid_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/dev.jsonl'))
test_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/test.jsonl'))


vocab = vocabulary.Vocabulary(
    training_data["sentences"], training_data["labels"], False)


train_dataset = biodataset.BioDataset(
    training_data["sentences"], training_data['labels'], vocab.word_to_idx, vocab.labels_to_idx)

valid_dataset = biodataset.BioDataset(
    valid_data["sentences"], valid_data['labels'], vocab.word_to_idx, vocab.labels_to_idx)

test_dataset = biodataset.BioDataset(
    test_data["sentences"], test_data['labels'], vocab.word_to_idx, vocab.labels_to_idx)


train_dataloader = DataLoader(
    train_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn, shuffle=False)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=utils.BATCH_SIZE, collate_fn=utils.collate_fn, shuffle=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, collate_fn=utils.collate_fn, shuffle=False)
print(len(vocab.word_to_idx))
#Uncomment the following lines and comment "embedding = None" to load a pretrained embedding

# model = torch.load((os.path.join(
#            utils.DIRECTORY_NAME, '0.702--0.721.pt')))
#embedding = model['word_embeddings.weight']
embedding = None

model = bio.BioClassifier(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
                          len(vocab.word_to_idx), len(vocab.labels_to_idx), utils.LAYERS_NUM, device, embedding)
loss_function = nn.CrossEntropyLoss(ignore_index=vocab.labels_to_idx["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=utils.LEARNING_RATE)

trainer = tr.Trainer(model, optimizer, device,
                     loss_function, vocab.idx_to_labels)
logs = trainer.train(train_dataloader, valid_dataloader, utils.EPOCHS_NUM)
utils.plot_logs(logs, 'Train vs Valid loss plus F1')
results = trainer.test(test_dataloader, torch.load(os.path.join(
    utils.DIRECTORY_NAME, '0.702--0.721.pt')))
print(results)
