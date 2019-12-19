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
batchSize = 32

trainDir = '../data/oxford-flowers17/train'
validDir = '../data/oxford-flowers17/val'
testDir = '../data/oxford-flowers17/test'

imTransform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainData = ImageFolder(trainDir, transform = imTransform)
trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = batchSize, shuffle = True)

validData = ImageFolder(validDir, transform = imTransform)
validDataLoader = torch.utils.data.DataLoader(validData, batch_size = batchSize, shuffle = False)

testData = ImageFolder(testDir, transform = imTransform)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size = batchSize, shuffle = False)

model = torchvision.models.squeezenet1_1(pretrained=True).to(device)
#print (model.classifier)
model.classifier[1] = nn.Conv2d(512, 17, 1, 1).to(device)
for param in model.parameters() :
	param.requires_grad = False
for param in model.classifier.parameters() :
	param.requires_grad= True
#print (model)

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
		diffArray = (classOut - outData)
		diffCopy = diffArray.cpu()
		accurateSample = np.where(diffCopy == 0)
		numAccurate = accurateSample[0].size
		testAccuracy += numAccurate

testAccuracy/= (step * batchSize)
print (testAccuracy)
