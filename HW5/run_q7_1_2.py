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

batchSize = 16
imTransform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.13,), (0.30,))])
trainData = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = imTransform)
trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = batchSize, shuffle = True)
testData = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = imTransform)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size = batchSize, shuffle = False)

class Net(nn.Module) :
	def __init__(self, numChannelsIn, numClasses) :
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = numChannelsIn, out_channels = 4, kernel_size = 3, stride = 1, bias = True)
		self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 2, stride = 1, bias = True)
		self.fc1 = nn.Linear(288, 32)
		self.fc2 = nn.Linear(32, numClasses)

	def forward(self, x) :
		x = self.conv1(x)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.dropout(F.relu(x), p = 0.2)
		x = self.fc2(x)
		return x

model = Net(1, 10).to(device)
numEpochs = 30
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
		#print(outData)
		modelOut = model(inData)
		modelParameters = list(model.parameters())

		loss = F.cross_entropy(modelOut, outData)
		epochLoss += loss
		optimizer = optim.SGD(modelParameters, lr = learning_rate)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		classOut = torch.max(modelOut, 1)[1]
		diffArray = classOut - outData
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
plt.legend(['loss', 'accuracy'])
plt.show()

testAccuracy = 0
with torch.no_grad() :
	for step, sample in enumerate(testDataLoader) :
		inData = sample[0].to(device)
		outData = sample[1].to(device)
		modelOut = model(inData)
		classOut = torch.max(modelOut, 1)[1]
		diffArray = classOut - outData
		diffCopy = diffArray.cpu()
		accurateSample = np.where(diffCopy == 0)
		numAccurate = accurateSample[0].size
		testAccuracy += numAccurate

testAccuracy/= (step * batchSize)
print (testAccuracy)
