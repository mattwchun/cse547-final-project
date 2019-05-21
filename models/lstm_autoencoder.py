#!/usr/bin/env python
# coding: utf-8
#
# Adapted from https://github.com/shuuchen/video_autoencoder


import torch
import torch.nn as nn
from torchvision import transforms, datasets
from resnet_feature_extracter import Img2Vec
import numpy as np
import os
import time
import copy
from PIL import Image

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### Hyper parameters


sequence_length = 10
#input_size = 2048
input_size = 512
hidden_size = 32#64#1024
num_layers = 2
batch_size = sequence_length # set to the number of images of a seqence # 36
num_epoches = 100#250
learning_rate = 0.01


# ### Feature vector extractor

extractor = Img2Vec()


# ### Antoencoder definition


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, bidirectional)

    def forward(self, x):
        encoded_x = self.encoder(x).expand(-1, sequence_length, -1)
        decoded_x = self.decoder(encoded_x)

        return (decoded_x, encoded_x[:, -1, :])


# ### Data preparation

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



# In[28]:


'''
data_dir = './pregnant'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          transform=data_transforms) for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                               batch_size=batch_size, shuffle=False) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
'''
# data_dir = './pregnant'
data_dir = '../data'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
# data_transforms = transforms.ToTensor()

# image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
#                                           transform=data_transforms) for x in ['data']}
image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x)) for x in ['all']}

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                               batch_size=batch_size, shuffle=False) for x in ['all']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['all']}

# print(image_datasets)
# print(data_loaders)
# print(dataset_sizes)


# ### Train the model

def train_model(model, criterion, optimizer, num_epoches=25):
    #losses = {'train': [], 'val': []}
    losses = {'all': []}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epoches):
        print('Epoch {} / {}'.format(epoch + 1, num_epoches))
        print('-' * 10)

        #for phase in ['train', 'val']:
        for phase in ['all']:
            if phase == 'val':
                # scheduler.step()
                model.eval()
            else:
                model.train()

            running_loss = 0.0

            allEmbeddings = np.array([])

            # sequence input
            for i in range(0, dataset_sizes['all'] - batch_size + 1, batch_size):
                imgs = []
                for j in range(i, i + batch_size):
                    path = os.path.join('{}{}.jpg'.format('../data/all/data/', j))
                    # img = Image.open(path)
                    img = data_transforms(Image.open(path))
                    imgs.append(img)
                inputs = torch.stack(imgs)
            #for inputs, _, paths in data_loaders[phase]:
                inputs = extractor.get_vec(inputs)

                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase != 'val'):
                    outputs, embeddings = model(inputs)
                    embeddingAsList = embeddings[0].cpu().detach().numpy()
                    if i == 0:
                        allEmbeddings = embeddingAsList
                    else:
                        allEmbeddings = np.vstack((allEmbeddings, embeddingAsList))

                    inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
                    loss = criterion(outputs, inputs[:, inv_idx, :])

                    if phase != 'val':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            losses[phase].append(epoch_loss)

            #print('{} Loss: {:4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if epoch == num_epoches - 1:
                # output only the last
                np.savetxt("embeddings.csv", allEmbeddings, delimiter=",")



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, losses


model = AutoEncoderRNN(input_size, hidden_size, num_layers)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model, losses = train_model(model, criterion, optimizer, num_epoches)
torch.save(model.state_dict(), './lstm_autoencoder_model.pt')


# ### Plot training/val curves

plt.figure(figsize=(20, 10))

#for phase in ['train', 'val']:
for phase in ['all']:
    plt.plot(losses[phase], label='{} loss'.format(phase))

plt.legend()

plt.title('train/val losses')

plt.xlabel('iteration')
plt.ylabel('loss')

plt.grid(True)
plt.show()
