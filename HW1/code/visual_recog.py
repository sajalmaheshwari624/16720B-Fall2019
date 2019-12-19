import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
import matplotlib.pyplot as plt
from PIL import Image

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    for i, files in enumerate(train_data['files']) :
        filePath = '../data/' + files
        if i % 5 == 0 :
            print (i)
        if i == 0 :
            train_histogram = get_image_feature(filePath, dictionary, 3, 100)
            labels = train_data['labels'][i]
        else :
            train_histogram = np.vstack((train_histogram, get_image_feature(filePath, dictionary, 3, 100)))
            labels = np.append(labels, train_data['labels'][i])
    outFile = 'trained_system.npz'
    np.savez(outFile,  features = train_histogram, labels = labels, dictionary = dictionary, SPM_layer_num = 3)




def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    word_dic = trained_system['dictionary']
    trained_hist = trained_system['features']
    trained_labels = trained_system['labels']
    num_layers = trained_system['SPM_layer_num']
    C = np.zeros((8,8))
    for i, files in enumerate(test_data['files']) :
        filePath = '../data/' + files
        test_hist = get_image_feature(filePath, word_dic, num_layers, 100)
        distVec = distance_to_set(test_hist, trained_hist)
        closest_im = np.argmax(distVec)
        closest_label = trained_labels[closest_im]
        true_label = test_data['labels'][i]
        print (i, true_label, closest_label)
        C[true_label][closest_label] += 1
        '''
        if true_label == 3 and closest_label == 7 :
            originalIm = Image.open(filePath)
            originalIm = np.array(originalIm) / 255
            plt.imshow(originalIm)
            plt.show()
        '''
    print (C)
    accuracy = C.trace() / np.sum(C[:])
    print (accuracy)


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----
    inImage = Image.open(file_path)
    inImage = np.array(inImage) / 255
    wordmap = visual_words.get_visual_words(inImage, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return feature


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    trainingImages = histograms.shape[0]
    word_hist = np.reshape(word_hist, (1, word_hist.shape[0]))
    word_hist = np.repeat(word_hist, trainingImages, axis = 0)
    histDiff = histograms - word_hist
    ansMat = np.zeros(histograms.shape)
    ansMat[histDiff < 0] = histograms[histDiff < 0]
    ansMat[histDiff >= 0] = word_hist[histDiff >= 0]
    ansVec = np.sum(ansMat, axis = 1)
    return ansVec



def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    bins = bins=np.arange(0, dict_size + 1)
    hist,_ = np.histogram(wordmap, bins = bins)
    hist = hist / np.sum(hist)
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    weights = np.array([0.5])
    partitions = np.array([1])
    for i in range(layer_num - 1) :
        partitions = np.append(partitions, 2 * partitions[-1])
        if i == layer_num - 2:
            weights = np.append(weights, weights[-1])
        else :
            weights = np.append(weights, weights[-1]/2)
    axisZeroLen = partitions[-1] * (wordmap.shape[0] // partitions[-1])
    axisOneLen = partitions[-1] * (wordmap.shape[1] // partitions[-1])
    wordmap = wordmap[0:axisZeroLen, 0:axisOneLen]
    wordMapPartitionsH = np.hsplit(wordmap, partitions[-1])
    wordMapPartitions = []
    for horizontalPartition in wordMapPartitionsH :
        np.vsplit(horizontalPartition, partitions[-1])
        wordMapPartitions.extend(np.vsplit(horizontalPartition, partitions[-1]))
    finest_vectors = []
    for imageParts in wordMapPartitions :
        finest_vectors.append(get_feature_from_wordmap(imageParts, 100))
    feature_vectors = [finest_vectors]
    for i in range(layer_num - 1) :
        partitionToSum = partitions[-1] * partitions[-1] / (partitions[layer_num - 2 - i] * partitions[layer_num - 2 - i])
        #print (partitionToSum)
        coarseVector = []
        prevj = 0
        for j in range(int(partitionToSum), int(partitions[-1] * partitions[-1] + 1), int(partitionToSum)) :
            finest_vectors_slice = finest_vectors[prevj:j]
            finest_vectors_slice_sum = sum(finest_vectors_slice) 
            coarseVector.append(finest_vectors_slice_sum)
            prevj = j
        feature_vectors.append(coarseVector)
    output_feature = np.array([])
    for i in range(len(feature_vectors)) :
        scale_feature = feature_vectors[i]
        for j in range(len(scale_feature)) :
            final_feature = scale_feature[j]
            final_feature = (final_feature / np.sum(final_feature)) * weights[len(weights) - 1 - i]
            output_feature = np.append(output_feature, final_feature[:])
    return output_feature / np.sum(output_feature)





train_data = np.load("../data/train_data.npz")
inImagePath = '../data/' + train_data['files'][0]
'''
inImage = Image.open(inImagePath)
inImage = np.array(inImage) / 255
wordmap = visual_words.get_visual_words(inImage, 'dictionary.npy')
word_hist = get_feature_from_wordmap_SPM(wordmap, 3, 100)
histograms = np.load('histograms.npy')
minImageIndex = distance_to_set(word_hist, histograms)
print (minImageIndex)
outImagePath = '../data/' + train_data['files'][minImageIndex]
outImage = Image.open(outImagePath)
outImage = np.array(outImage) / 255
fig, axes = plt.subplots(2,1)
axes[0].imshow(inImage)
axes[1].imshow(outImage)
plt.show()
'''


'''
train_data = np.load("../data/train_data.npz")
print (train_histogram.shape)
#outArrays = np.hsplit(wordmap, 16)
#print (outArrays[0].shape)
'''

#build_recognition_system()
#evaluate_recognition_system()


