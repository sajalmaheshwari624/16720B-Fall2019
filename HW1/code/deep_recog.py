import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
from torchvision.utils import save_image
from PIL import Image
import scipy
import matplotlib.pyplot as plt

def build_recognition_system(vgg16, num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N, K)
	* labels: numpy.ndarray of shape (N)
	'''
	train_data = np.load("../data/train_data.npz")
	for i, files in enumerate(train_data['files']) :
		filePath = '../data/' + files
		print (i)
		if i == 0 :
			train_histogram = get_image_feature([i, filePath, vgg16])
			print (train_histogram)
			labels = train_data['labels'][i]
		else :
			bufferHist =  get_image_feature([i, filePath, vgg16])
			print (bufferHist)
			train_histogram = np.vstack((train_histogram, get_image_feature([i, filePath, vgg16])))
			labels = np.append(labels, train_data['labels'][i])
	outFile = 'trained_system_deep.npz'
	#np.savez(outFile,  features = train_histogram, labels = labels)

	

def evaluate_recognition_system(vgg16, num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8, 8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system_deep.npz")
	C = np.zeros((8,8))
	for i, files in enumerate(test_data['files']) :
		filePath = '../data/' + files
		test_hist = get_image_feature([i, filePath, vgg16])
		distVec = distance_to_set(test_hist, trained_system['features'])
		closest_im = np.argmin(distVec)
		closest_label = trained_system['labels'][closest_im]
		true_label = test_data['labels'][i]
		print (i, true_label, closest_label)
		C[true_label][closest_label] += 1

		'''
		Validate implementation of convolution layers in Q 3.1
		if i == 0 :
			test_hist = get_image_feature([i, filePath, vgg16])
			im = preprocess_image(filePath)
			im_manual = np.ones([224,224,3])
			im_manual[:,:,0] = im[0,:,:].numpy()
			im_manual[:,:,1] = im[1,:,:].numpy()
			im_manual[:,:,2] = im[2,:,:].numpy()
			weights = util.get_VGG16_weights()
			test_hist_manual = network_layers.extract_deep_feature(im_manual, weights)
			diff = abs(test_hist - test_hist_manual)
			print (np.sum(diff))
			#print (np.transpose(diff))
			break
		'''
	print (C)
	accuracy = C.trace() / np.sum(C[:])
	return C



def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H, W, 3)

	[output]
	* image_processed: torch.Tensor of shape (3, H, W)
	'''

	# ----- TODO -----
	inImage = Image.open(image)
	inImage = inImage.resize((224, 224))
	inImage = np.array(inImage) / 255
	mean = np.array([0.485,0.456,0.406])
	std = np.array([0.229,0.224,0.225])
	inImage[:,:,0] = (inImage[:,:,0] - mean[0]) / std[0]
	inImage[:,:,1] = (inImage[:,:,1] - mean[1]) / std[1]
	inImage[:,:,2] = (inImage[:,:,2] - mean[2]) / std[2]
	tImage = torch.ones(3, 224, 224).double()
	tImage[0,:,:] = torch.from_numpy(inImage[:,:,0])
	tImage[1,:,:] = torch.from_numpy(inImage[:,:,1])
	tImage[2,:,:] = torch.from_numpy(inImage[:,:,2])
	return tImage



def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	
	[output]
	* feat: evaluated deep feature
	'''

	i, image_path, vgg16 = args

	# ----- TODO -----
	new_model = vgg16
	new_classifier = torch.nn.Sequential(*list(new_model.classifier.children())[:4])
	new_features = torch.nn.Sequential(*list(new_model.features.children()))[0]
	new_model.classifier = new_classifier
	pre_processed_image = preprocess_image(image_path)
	pre_processed_tensor = pre_processed_image.unsqueeze(0)
	out = new_model(pre_processed_tensor)
	out = out.detach().numpy()
	return out





def distance_to_set(feature, train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N, K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	# ----- TODO -----
	distances = scipy.spatial.distance.cdist (feature, train_features)
	return distances
