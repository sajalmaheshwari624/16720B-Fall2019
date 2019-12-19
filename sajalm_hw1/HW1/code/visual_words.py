import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import math
import tempfile
import subprocess
from PIL import Image

def filterIm(im, filterType, scale) :
    if im.shape[2] == 1 :
        im = np.stack((im, im, im), axis=-1)
    if filterType == 'Gauss' :
        outIm1 = scipy.ndimage.gaussian_filter(im[:,:,0], scale);
        outIm2 = scipy.ndimage.gaussian_filter(im[:,:,1], scale);
        outIm3 = scipy.ndimage.gaussian_filter(im[:,:,2], scale);
        outIm = np.stack((outIm1, outIm2, outIm3), axis=-1)
    elif filterType == 'GaussX' :
        outIm1 = scipy.ndimage.gaussian_filter(im[:,:,0], scale, order = [1,0]);
        outIm2 = scipy.ndimage.gaussian_filter(im[:,:,1], scale, order = [1,0]);
        outIm3 = scipy.ndimage.gaussian_filter(im[:,:,2], scale, order = [1,0]);
        outIm = np.stack((outIm1, outIm2, outIm3), axis=-1)
    elif filterType == 'GaussY' :
        outIm1 = scipy.ndimage.gaussian_filter(im[:,:,0], scale, order = [0,1]);
        outIm2 = scipy.ndimage.gaussian_filter(im[:,:,1], scale, order = [0,1]);
        outIm3 = scipy.ndimage.gaussian_filter(im[:,:,2], scale, order = [0,1]);
        outIm = np.stack((outIm1, outIm2, outIm3), axis=-1)
    else :
        outIm1 = scipy.ndimage.gaussian_laplace(im[:,:,0], scale);
        outIm2 = scipy.ndimage.gaussian_laplace(im[:,:,1], scale);
        outIm3 = scipy.ndimage.gaussian_laplace(im[:,:,2], scale);
        outIm = np.stack((outIm1, outIm2, outIm3), axis=-1) 
    return outIm

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----
    inputIm = skimage.color.rgb2lab(image)
    filterTypes = ['Gauss', 'LoG', 'GaussY', 'GaussX']
    scaleTypes = np.array([1.0, 2.0, 4.0, 8.0, 8 * math.sqrt(2)])
    count = 0
    for scales in scaleTypes :
        for filters in filterTypes :
            count += 1
            #print (count)
            if (count == 1) :
                filterResponse = filterIm(inputIm, filters, scales)
                #print (np.shape(filterResponse))
            else :
                filterResponse = np.concatenate((filterResponse, filterIm(inputIm, filters, scales)), axis=2)
                #print (filterResponse.shape)
    #util.display_filter_responses(filterResponse)
    return filterResponse

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    inResponse = extract_filter_responses(image)
    #print (inResponse.shape)
    dictionaryMat = dictionary
    imageMat = np.reshape(inResponse, (inResponse.shape[0]*inResponse.shape[1], inResponse.shape[2]))
    #imageMat = np.reshape(inImage, (inImage.shape[0], inImage.shape[1], inImage.shape[2]))
    #print (imageMat.shape)
    distanceMat = scipy.spatial.distance.cdist(imageMat, dictionaryMat, metric = 'euclidean')
    wordmap = np.argmin(distanceMat, axis = 1)
    wordmap = np.reshape(wordmap, (inResponse.shape[0], inResponse.shape[1]))
    return wordmap

def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''

    # ----- TODO -----
    i, alpha, image_path = args
    #print (i)
    inputIm = Image.open(image_path)
    inputIm = np.array(inputIm) / 255
    #print (inputIm.shape)
    response = extract_filter_responses(inputIm)
    responseShape = response.shape
    height = responseShape[0]
    width = responseShape[1]
    depth = responseShape[2]
    randPerm = np.random.permutation(height * width)
    randPerm = randPerm[0:alpha]
    responseReshape = np.reshape(response, (height * width, -1))
    alphaResponse = responseReshape[randPerm]
    #currentDir = os.getcwd()
    #tempOutFile = tempfile.TemporaryFile(dir = currentDir)
    #np.save(tempOutFile, alphaResponse)
    return alphaResponse

    #print (str(i))
    #print (alphaResponse.shape, alphaResponse)
    #currentDir = os.getcwd()
    #commonName = 'tempResponse_'
    #imageName = str(i)
    #tempOutDir = tempfile.TemporaryDirectory(dir = currentDir)
    #print (tempOutDir.name)
    #tempOutFile = tempfile.NamedTemporaryFile(prefix = commonName, dir = tempDirName, suffix = imageName, delete=False)
    #print(tempOutFile.name)
    #print('Created temporary directory:', '../tempResponses')
        #tempOutFile = tempfile.TemporaryFile(suffix="_" + str(i))
        #print("Name of the file is:", tempOutFile.name)
        #np.save(tempOutFile, alphaResponse)
        #tempOutFile.seek(0)
        #out = np.load(tempOutFile)
        #print (out)



def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''
    train_data = np.load("../data/train_data.npz")
    for i, files in enumerate(train_data['files']) :
        print (i)
        if i == 0 :
            train_matrix = compute_dictionary_one_image([i, 100, '../data/' + files])
            #print (train_matrix.shape)
        else :
            train_matrix = np.concatenate((train_matrix, compute_dictionary_one_image([i, 50, '../data/' + files])), axis = 0)
            #print (train_matrix.shape)
    print (train_matrix.shape)
    kmeans = sklearn.cluster.KMeans(n_clusters = 100).fit(train_matrix)
    dictionary = kmeans.cluster_centers_
    print (dictionary)
    np.save('dictionary', dictionary)
        #subprocess.call(compute_dictionary_one_image(), *, stdin=None, stdout=None, stderr=None, shell=False)
    # ----- TODO -----
    
    pass

#compute_dictionary()
'''
inImagePath = '../data/park/labelme_djdpufqmxpltzcp.jpg'
inImage = Image.open(inImagePath)
inImage = np.array(inImage) / 255
fig, axes = plt.subplots(2,1)
axes[0].imshow(inImage)
    #print (inputIm.shape)
wordmap = get_visual_words(inImage, 'dictionary.npy')
axes[1].imshow(wordmap)
plt.show()
'''
#compute_dictionary_one_image([1, 100, '../data/aquarium/sun_aztvjgubyrgvirup.jpg'])
#compute_dictionary_one_image([2, 100, '../data/aquarium/sun_aztvjgubyrgvirup.jpg'])


