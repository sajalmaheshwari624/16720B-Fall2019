import torch
import torchvision
import scipy.io
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
learning_rate = 0.01
batchSize = 32

trainDir = '../data/oxford-flowers17/train'
validDir = '../data/oxford-flowers17/val'
testDir = '../data/oxford-flowers17/test'

imTransform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize((0.42,0.42,0.28), (0.07,0.64,0.06))])

trainData = ImageFolder(trainDir, transform = imTransform)
trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = batchSize, shuffle = True)

validData = ImageFolder(validDir, transform = imTransform)
validDataLoader = torch.utils.data.DataLoader(validData, batch_size = batchSize, shuffle = False)

testData = ImageFolder(testDir, transform = imTransform)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size = batchSize, shuffle = False)

class Net(nn.Module) :
	def __init__(self, numChannelsIn, numClasses) :
		super(Net, self).__init__()

		self.conv1 	= nn.Conv2d(in_channels = numChannelsIn, 	out_channels = 8, 	kernel_size = 7, stride = 2, padding = 3, bias = True)
		self.conv2 	= nn.Conv2d(in_channels = 8, 	out_channels = 16, 	kernel_size = 5, stride = 2, padding = 2, bias = True)		
		self.conv3 	= nn.Conv2d(in_channels = 16, 	out_channels = 128, kernel_size = 5, stride = 2, padding = 2, bias = True)
		self.fc1 = nn.Linear(1152, 64)
		self.fc2 = nn.Linear(64, numClasses)

	def forward(self, inImage) :
		
		conv1Out = F.max_pool2d(F.relu(self.conv1(inImage)),2)
		conv2Out = F.max_pool2d(F.relu(self.conv2(conv1Out)),2)
		conv3Out = F.max_pool2d(F.relu(self.conv3(conv2Out)),2)

		x = torch.flatten(conv3Out, 1)
		#print (x.shape)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x


model = Net(3, 17).to(device)
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
	print (epochLoss, epochAccuracy)
	trainLossArray[epoch] = epochLoss
	trainAccuArray[epoch] = epochAccuracy

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


