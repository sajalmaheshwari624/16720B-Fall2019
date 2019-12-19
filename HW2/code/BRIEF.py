import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                                patch and are each (nbits,) vectors. 
    '''
    
    #############################
    # TO DO ...
    # Generate testpattern here
    '''
    mean = patch_width*patch_width // 2
    cov = patch_width*patch_width / 5

    testPairsX = np.round(np.random.normal(mean, cov, nbits))
    testPairsX[testPairsX > patch_width*patch_width-1] = patch_width-1
    testPairsX[testPairsX < 0] = 0
    compareX = testPairsX.astype(int)

    testPairsY = np.round(np.random.normal(mean, cov, nbits))
    testPairsY[testPairsY > patch_width*patch_width-1] = patch_width-1
    testPairsY[testPairsY < 0] = 0
    compareY = testPairsY.astype(int)
    '''
    compareX = np.random.randint(0, patch_width**2 - 1, nbits)
    compareY = np.random.randint(0, patch_width**2 - 1, nbits)
    return  compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])


def computeBrief(im, compareX, compareY, gaussian_pyramid, locsDoG, k=0, levels=0):
    '''
    Compute brief feature
    INPUT
        locsDoG - locsDoG are the keypoint locations returned by the DoG
                detector.
        levels  - Gaussian scale levels that were given in Section1.
        compareX and compareY - linear indices into the 
                                (patch_width x patch_width) image patch and are
                                each (nbits,) vectors.
    
    
    OUTPUT
        locs - an m x 3 vector, where the first two columns are the image
                coordinates of keypoints and the third column is the pyramid
                level of the keypoints.
        desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
                of valid descriptors in the image and will vary.
    '''
    
    ##############################
    # TO DO ...
    # compute locs, desc here
    locsX = (locsDoG[:,1])
    locsY = (locsDoG[:,0])
    #print (locsX, locsY)
    locsXValidHigh = np.where((locsX < im.shape[0] - 9/2 - 1))
    locsXValidLow = np.where((locsX >= 9//2))
    locsXvalid = np.intersect1d(locsXValidHigh, locsXValidLow)

    locsYValidHigh = np.where((locsY < im.shape[1] - 9/2 - 1))
    locsYValidLow = np.where((locsY >= 9//2))
    locsYvalid = np.intersect1d(locsYValidHigh, locsYValidLow)

    locsValid = np.intersect1d(locsXvalid, locsYvalid)
    locs = locsDoG[locsValid]
    locs2D = locs[:,0:2]
    desc = np.zeros([len(locs2D),len(compareX)])
    count = 0
    for indices in locs2D :
        imPatch = im[indices[1] - 9//2 : indices[1] + 9 // 2 + 1, indices[0] - 9 // 2 : indices[0] + 9 // 2 + 1]
        #print (imPatch.shape)
        #cv2.imshow('Jai matadi!', imPatch)
        #cv2.waitKey(0)
        imPatch = np.reshape(imPatch, (9*9,1))
        imX = imPatch[compareX]
        imY = imPatch[compareY]
        imDiff = imX - imY
        imDiff[imDiff < 0] = 0
        imDiff[imDiff > 0] = 1
        desc[count,:] = imDiff[:,0]
        count+= 1
    return locs, desc



def briefLite(im):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    
    ###################
    # TO DO ...
    locsDoG, gaussian_pyramid = DoGdetector(im)
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    compareX, compareY = np.load('../results/testPattern.npy')
    locs, desc = computeBrief(im, compareX, compareY, gaussian_pyramid, locsDoG)
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()    
    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    
    # test briefLite
    '''
    im = cv2.imread('../data/model_chickenbroth.jpg')
    if len(im.shape)==3 :
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    a = computeBrief(im,compareX, compareY)
    locs, desc = briefLite(im)  
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
    
    '''
    # test matches
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    #../data/chickenbroth_01.jpg
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)


    #cv2.imshow('1',im1)
    #cv2.imshow('2',im2)
    #cv2.waitKey(0)
    
    plotMatches(im1,im2,matches,locs1,locs2)