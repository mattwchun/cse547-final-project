# Note:
#
# Following a model similar to the LSTM Autoencoder model from:
# http://arxiv.org/abs/1502.04681

# TODO:
# Turn each trailer into a fixed size number of frames (downsample)
# * Normalize photos (scale -> take center) before ResNet
# * Create data loader
# * create train and validation set of movies (optional can overfit)
# extract feature rep for each movie
# train

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from img_to_vec import Img2Vec
from PIL import Image
from torchvision.datasets import ImageFolder
import glob

img2vec = Img2Vec()

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, fc_o_size = 32, lstm_h_size = 128, lstm_num_layers = 1):
        super(LSTMAutoencoder, self).__init__()

        self.input_size = input_size
        self.fc_o_size = fc_o_size
        self.lstm_h_size = lstm_h_size
        self.lstm_num_layers = lstm_num_layers

        # Encoder
        self.lstm_enc = nn.LSTM(input_size, hidden_size=lstm_h_size, num_layers = lstm_num_layers)
        self.fc_enc = nn.Linear(lstm_h_size, fc_o_size)

        # Decoder
        self.lstm_dec = nn.LSTM(fc_o_size, hidden_size=lstm_h_size, num_layers = lstm_num_layers)
        self.fc_dec = nn.Linear(lstm_h_size, input_size)

    # input = (number of frames for a movie, size of batch or number of movie trailers, 512) = (F, M, 512)
    # input_reverse = (F, reversed M, 512)
    def forward(self, input):
        # hn = torch.randn(1, input.size(1), self.lstm_h_size).cuda()
        # cn = torch.randn(1, input.size(1), self.lstm_h_size).cuda()
        print(input.size(1))
        hn = torch.randn(10, 1, input.size(1))
        cn = torch.randn(10, 1, input.size(1))

        output, (hn, cn) = self.lstm_enc(input, (hn, cn))
        output = self.fc_enc(output)

        # extract last vector of each sequence from fc enc layer
        featureRep = output[:, -1, :]

        output, (hn, cn) = self.lstm_dec(output, (hn, cn))
        output = self.fc_enc(output)
        return (output, featureRep)

def train(model, optimizers, data, epochs=10):
    for name, optimizer in optimizers.items():
        print("Optimizer name: %s" % name)

        # split data into 2 sets train and validation (90/10)

        model.train()
        train_loss = []
        train_accu = []
        for epoch in range(epochs):
            for batchNum, moviesBatch in enumerate(data):
                img, _class = moviesBatch
                optimizer.zero_grad()
                print(img)
                print(_class)
                output, featureRep = model.forward((img))

                # img = Variable(img).cuda()
                img = Variable(img)
                loss = F.mse_loss(img,output)
                loss.backward()  # calc gradients
                train_loss.append(loss.data[0].item())
                optimizer.step()  # update gradients
                prediction = output.data.max(1)[1]  # first column has actual prob.

                print('epoch [%d/%d], loss:%.4f' % (epoch + 1, epochs, loss.data[0]))

    torch.save(model.state_dict(), './lstm_autoencoder.pth')

def loader_fn(filename):
    img = Image.open(filename)
    return img2vec.get_vec(img)

def isValidPic(path):
    with Image.open(path) as img:
        return img.verify()

def getDataLoader(dir, loader_fn):
    imgFolder = ImageFolder(root=dir, loader=loader_fn)
    return torch.utils.data.DataLoader(imgFolder,
                                             batch_size=10,
                                             num_workers=4)

def main():
    model = LSTMAutoencoder(512)
    optimizers = {'SGD': optim.SGD(model.parameters(), lr=0.01)}
    data = getDataLoader('../data/frames', loader_fn)

    train(model, optimizers, data)

main()
