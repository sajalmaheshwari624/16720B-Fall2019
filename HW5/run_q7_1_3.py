import torch
import torchvision
import scipy.io
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from q4 import *
import os

device = torch.device("cuda:0")
batchSize = 32

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_data_examples = train_data['train_data']
train_data_examples = np.array([np.reshape(x, (1, 32, 32)) for x in train_data_examples])
train_data_labels = train_data['train_labels']

test_data_examples = test_data['test_data']
test_data_examples = np.array([np.reshape(x, (1, 32, 32)) for x in test_data_examples])
test_data_labels = test_data['test_labels']


train_data_torch = torch.as_tensor(train_data_examples, dtype=torch.float)
test_data_torch = torch.as_tensor(test_data_examples, dtype=torch.float)

train_label_torch = torch.as_tensor(train_data_labels, dtype=torch.float)
test_label_torch = torch.as_tensor(test_data_labels, dtype=torch.float)

trainDataLoad = data.TensorDataset(train_data_torch, train_label_torch)
testDataLoad = data.TensorDataset(test_data_torch, test_label_torch)
trainDataLoader = data.DataLoader(trainDataLoad, batch_size = batchSize, shuffle = True)
testDataLoader = data.DataLoader(testDataLoad, batch_size = batchSize, shuffle = False)

class Net(nn.Module) :
	def __init__(self, numChannelsIn, numClasses) :
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = numChannelsIn, out_channels = 8, kernel_size = 3, stride = 1, bias = True)
		self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, bias = True)
		self.fc1 = nn.Linear(12544, 1024)
		self.fc2 = nn.Linear(1024, numClasses)

	def forward(self, x) :
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.dropout(F.relu(x), p = 0.2)
		x = self.fc2(x)
		return x

'''
class Net(nn.Module):
    def __init__(self, numChannelsIn, numClasses):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(numChannelsIn, 8, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(16*16*16, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, numClasses))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 16*16*16)
        x = self.fc1(x)
        return x
'''

model = Net(1, 36).to(device)
numEpochs = 50
learning_rate = 0.01

trainLossArray = np.zeros(numEpochs)
trainAccuArray = np.zeros(numEpochs)
for epoch in range(numEpochs) :
	epochLoss = 0
	epochAccuracy = 0
	print (epoch)
	for step, sample in enumerate(trainDataLoader) :
		loss = torch.zeros(1).to(device)
		loss.requires_grad = False

		inData 	= sample[0].to(device)
		outData = sample[1].to(device)
		modelOut = model(inData)
		modelParameters = list(model.parameters())
		targets = torch.max(outData, 1)[1]
		loss = F.cross_entropy(modelOut, targets)
		epochLoss += loss
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		classOut = torch.max(modelOut, 1)[1]
		diffArray = classOut - targets
		diffCopy = diffArray.cpu()
		accurateSample = np.where(diffCopy == 0)
		numAccurate = accurateSample[0].size
		epochAccuracy += numAccurate

	epochLoss/= step
	epochAccuracy/= (step * batchSize)
	trainLossArray[epoch] = epochLoss
	trainAccuArray[epoch] = epochAccuracy
	print (epochLoss, epochAccuracy)

plt.plot(trainLossArray)
plt.plot(trainAccuArray)
plt.show()


with torch.no_grad() :
	testAccuracy = 0
	for step, sample in enumerate(testDataLoader) :
		inData = sample[0].to(device)
		outData = sample[1].to(device)
		modelOut = model(inData)
		classOut = torch.max(modelOut, 1)[1]
		targets = torch.max(outData, 1)[1]
		diffArray = classOut - targets
		diffCopy = diffArray.cpu()
		accurateSample = np.where(diffCopy == 0)
		numAccurate = accurateSample[0].size
		testAccuracy += numAccurate

testAccuracy/= (step * batchSize)
print (testAccuracy)
