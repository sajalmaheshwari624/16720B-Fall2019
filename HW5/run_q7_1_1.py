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

device = torch.device("cpu")
class Net(nn.Module) :
	def __init__(self, inputSize, outputSize) :
		super(Net, self).__init__()
		self.fc1 = nn.Linear(inputSize,64)
		self.fc2 = nn.Linear(64, outputSize)
	def forward (self, x) :
		x = self.fc1(x)
		x = F.sigmoid(x)
		x = self.fc2(x)
		return x


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_data_examples = train_data['train_data']
train_data_labels = train_data['train_labels']

valid_data_examples = valid_data['valid_data']
valid_data_labels = valid_data['valid_labels']

test_data_examples = test_data['test_data']
test_data_labels = test_data['test_labels']

train_data_torch = torch.as_tensor(train_data_examples, dtype=torch.float)
valid_data_torch = torch.as_tensor(valid_data_examples, dtype=torch.float)
test_data_torch = torch.as_tensor(test_data_examples, dtype=torch.float)

train_label_torch = torch.as_tensor(train_data_labels, dtype=torch.float)
valid_label_torch = torch.as_tensor(valid_data_labels, dtype=torch.float)
test_label_torch = torch.as_tensor(test_data_labels, dtype=torch.float)

shuffleFlagTrain = True
batchSize = 8
trainDataLoad = data.TensorDataset(train_data_torch, train_label_torch)
trainDataLoader = data.DataLoader(trainDataLoad, batch_size = batchSize, shuffle = shuffleFlagTrain)

shuffleFlagTest = False
testDataLoad = data.TensorDataset(test_data_torch, test_label_torch)
testDataLoader = data.DataLoader(testDataLoad, batch_size = batchSize, shuffle = shuffleFlagTest)

model = Net(1024, 36).to(device)
learning_rate = 0.08
numEpochs = 100
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
		#print(outData)
		targets = torch.max(outData, 1)[1]
		modelOut = model(inData)
		modelParameters = list(model.parameters())

		loss = F.cross_entropy(modelOut, targets)
		epochLoss += loss
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		classOut = torch.max(modelOut, 1)[1]
		diffArray = classOut - targets
		accurateSample = np.where(diffArray == 0)
		numAccurate = accurateSample[0].size
		epochAccuracy += numAccurate

	epochLoss/= step
	epochAccuracy/= train_data_labels.shape[0]
	trainLossArray[epoch] = epochLoss
	trainAccuArray[epoch] = epochAccuracy
	print (epochLoss, epochAccuracy)

plt.plot(trainLossArray)
plt.plot(trainAccuArray)
plt.legend(['loss', 'accuracy'])
plt.show()

testAccuracy = 0
with torch.no_grad() :
	for step, sample in enumerate(testDataLoader) :
		inData = sample[0].to(device)
		outData = sample[1].to(device)
		targets = torch.max(outData, 1)[1]
		modelOut = model(inData)
		classOut = torch.max(modelOut, 1)[1]
		diffArray = classOut - targets
		accurateSample = np.where(diffArray == 0)
		numAccurate = accurateSample[0].size
		testAccuracy += numAccurate

testAccuracy/= test_data_labels.shape[0]
print (testAccuracy) 