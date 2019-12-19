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
torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
imTransform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.13,), (0.30,))])
trainData = torchvision.datasets.EMNIST(root='./data', split = 'balanced', train = True, download = True, transform = imTransform)
trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = batchSize, shuffle = True)
testData = torchvision.datasets.EMNIST(root='./data', split = 'balanced', train = False, download = True, transform = imTransform)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size = batchSize, shuffle = False)

class Net(nn.Module) :
	def __init__(self, numChannelsIn, numClasses) :
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = numChannelsIn, out_channels = 16, kernel_size = 3, stride = 1, bias = True)
		self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 2, stride = 1, bias = True)
		self.fc1 = nn.Linear(1152, 64)
		self.fc2 = nn.Linear(64, numClasses)

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

model = Net(1, 47).to(device)
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
plt.show()

testAccuracy = 0
with torch.no_grad() :
	for step, sample in enumerate(testDataLoader) :
		inData = sample[0].to(device)
		#print (inData.shape)
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
	import string
	letterDict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
         20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
         30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
         40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}
	for img in os.listdir('../images'):
		charTensor = []
		dataMatrix = np.array([])
		im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
		bboxes, bw = findLetters(im1)
		clusterCenters = dict([])
		thresh = 150
		for index in range(bboxes.shape[0]) :
			box = bboxes[index,:]
			boxxCoord = (box[0] + box[2]) / 2

			count = 0
			for key, values in clusterCenters.items() :
				if abs(boxxCoord - key) > thresh :
					count += 1
				else :
					clusterCenters[key] = np.vstack((clusterCenters[key], np.array(box)))

			if count == len(clusterCenters) :
				clusterCenters[boxxCoord] = np.array(box)
		for center,entries in clusterCenters.items() :
			entriesTopX = entries[:,1]
			sortIndices = np.argsort(entriesTopX)
			sortEntries = np.array([])
			for sortedIndex in sortIndices :
				if sortEntries.size == 0 :
					sortEntries = entries[sortedIndex]
				else :
					sortEntries = np.vstack((sortEntries, entries[sortedIndex]))
			clusterCenters[center] = sortEntries

		kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
		breakIndices = np.array([])
		breakCount = 0
		for center, entries in clusterCenters.items() :
			bboxVal = entries
			for entry in entries :
				breakCount += 1
				imPatch = bw[entry[0]:entry[2], entry[1]:entry[3]]
				numRows = imPatch.shape[0]
				numCols = imPatch.shape[1]
				if numRows > numCols :
					diff = numRows - numCols
					imPatch = np.pad(imPatch, ((10,10),(diff // 2, diff // 2)), 'constant', constant_values=1)
				elif numCols > numRows :
					diff = numCols - numCols
					imPatch = np.pad(imPatch, ((10,10),(diff // 2, diff // 2)), 'constant', constant_values=1)
				imPatch = skimage.transform.resize(imPatch, (28, 28))
				imPatch = skimage.morphology.erosion(imPatch, kernel)
				imPatch = np.transpose(imPatch)
				imPatch = 1 - imPatch
				imTransform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.13,), (0.30,))])
				imPatchNorm = (imPatch - 0.13) / (0.30)
				#print (imPatchNorm.shape)
				imPathTorch = torch.as_tensor(np.reshape(imPatchNorm, (1,28,28)), dtype=torch.float)
				charTensor.append(imPathTorch)
			breakIndices = np.append(breakIndices, breakCount)
		dataTensor = torch.stack(charTensor, dim=0).to(device)
		#print (breakIndices)

		#print (imPathTorch.shape)
		modelOut = model(dataTensor)
		#print (modelOut.size())
		predLabels = torch.max(modelOut,1)[1]
		predLabels = predLabels.cpu()
		predLabels = predLabels.numpy()
		print (predLabels.size)
		ansString = ""
		for writeIndex in range(predLabels.size) :
			if writeIndex in breakIndices :
				print (ansString)
				ansString = ""
			ansString = ansString + letterDict[predLabels[writeIndex]]
		print (ansString)






