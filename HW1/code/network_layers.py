import numpy as np
import scipy.ndimage
import os
import util
import skimage
import matplotlib.pyplot as plt
import torchvision
import torch
from PIL import Image

def extract_deep_feature(x, vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H, W, 3)
	* vgg16_weights: list of shape (L, 3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	num_layers = len(vgg16_weights)
	count = 0
	count1 = 1
	for layerCount in range(0, num_layers) :
		layerID = vgg16_weights[layerCount][0]
		if layerID == 'conv2d' :
			x = multichannel_conv2d(x, vgg16_weights[layerCount][1], vgg16_weights[layerCount][2])
		elif layerID == 'relu' :
			x = relu(x)
		elif layerID == 'maxpool2d' :
			x = max_pool2d(x, vgg16_weights[layerCount][1])
		elif layerID == 'linear' :
			count += 1
			if count == 1 :
				x = np.transpose(x, (2,0,1)).flatten()
				x = linear(x, vgg16_weights[layerCount][1], vgg16_weights[layerCount][2])
			if count == 2 :
				x = linear(x, vgg16_weights[layerCount][1], vgg16_weights[layerCount][2])
				break
			#print (x.shape, vgg16_weights[layerCount][1].shape, vgg16_weights[layerCount][2].shape)
	return x

	#layerNames = 


def multichannel_conv2d(x, weight, bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H, W, input_dim)
	* weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H, W, output_dim)
	'''
	output_dim = weight.shape[0]
	input_dim = weight.shape[1]
	feat = np.zeros([x.shape[0], x.shape[1], output_dim])
	for j in range(output_dim) :
		yJNoBias = np.zeros([x.shape[0], x.shape[1]]) + bias[j]
		for k in range(input_dim) :
			xK = x[:,:,k]
			hjK = weight[j,k,:,:]
			convOutK = scipy.ndimage.convolve(xK, np.flipud(np.fliplr(hjK)), mode = 'constant', cval =0.0)
			yJNoBias = yJNoBias + convOutK
		feat[:,:,j] = yJNoBias
	return feat


def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	x[x < 0] = 0
	return x

def max_pool2d(x, size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H, W, input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size, W/size, input_dim)
	'''
	height = x.shape[0]
	width = x.shape[1]
	y = np.ones((height // size, width // size, x.shape[2]))
	#y = np.ones((height // size, width // size))
	countx = 0
	for i in range(0, height - 1, size) :
		county = 0
		for j in range (0, width - 1, size) :
			imPatch = x[i:i + size, j: j+ size, :]
			for k in range(0,imPatch.shape[2]) :
				y[countx, county, k] = np.max(imPatch[:,:,k])
			county += 1
		countx += 1
	return y


def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	return np.matmul(W,x) + b

'''
#path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
#inImage = Image.open(path_img)
inImage = np.ones([224,224,3])
vgg16 = torchvision.models.vgg16(pretrained=True).double()
vgg16.eval()
tImage = torch.ones(3, 224, 224).double()
tImage = tImage.unsqueeze(0)
new_features = torch.nn.Sequential(*list(vgg16.features.children()))[0]

c1 = new_features(tImage)
c2 = extract_deep_feature(inImage, util.get_VGG16_weights())
print (c1)
print (" ========= ========== ==========")
print (c2)
'''

