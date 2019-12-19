import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform
from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    dataMatrix = np.array([])
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    #print (bboxes)
    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    clusterCenters = dict([])
    thresh = 150
    for index in range(bboxes.shape[0]) :
        box = bboxes[index,:]
        boxxCoord = (box[0] + box[2]) / 2
        #if len(clusterCenters) == 0 :
            #clusterCenters[boxxCoord] = np.array(box)

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
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
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
                imPatch = np.pad(imPatch, ((diff // 2, diff // 2), (10,10)), 'constant', constant_values=1)
            imPatch = skimage.transform.resize(imPatch, (32, 32))
            imPatch = skimage.morphology.erosion(imPatch, kernel)
            imPatch = np.transpose(imPatch)
            imVector = imPatch.flatten()
            imVector = np.reshape(imVector, (1, imVector.size))
            if dataMatrix.size == 0 :
                dataMatrix = imVector
            else :
                dataMatrix = np.vstack((dataMatrix, imVector))
        breakIndices = np.append(breakIndices, breakCount)
    print (dataMatrix.shape)
    #print (breakIndices)
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    #print (letters)
    params = pickle.load(open('q3_weights.pickle','rb'))
    h1 = forward(dataMatrix,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    predLabels = np.argmax(probs, axis = 1)
    #print (predLabels.size)
    ansString = ""
    for writeIndex in range(predLabels.size) :
        if writeIndex in breakIndices :
            print (ansString)
            ansString = ""
        ansString = ansString + letters[predLabels[writeIndex]]
    print (ansString)