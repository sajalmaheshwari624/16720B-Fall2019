import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
                equation
    '''
    
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    # Write a homography operation such that P2 = H*P1
    HMat = np.zeros([2*p2.shape[1],9])
    HMat[range(1,2*p2.shape[1],2),0] = p1[0,:]
    HMat[range(1,2*p2.shape[1],2),1] = p1[1,:]
    HMat[range(1,2*p2.shape[1],2),2] = 1
    HMat[range(0,2*p2.shape[1],2),3] = -1*p1[0,:]
    HMat[range(0,2*p2.shape[1],2),4] = -1*p1[1,:]
    HMat[range(0,2*p2.shape[1],2),5] = -1
    HMat[range(0,2*p2.shape[1],2),6] = p2[1,:]*p1[0,:]
    HMat[range(1,2*p2.shape[1],2),6] = -1*p2[0,:]*p1[0,:]
    HMat[range(0,2*p2.shape[1],2),7] = p2[1,:]*p1[1,:]
    HMat[range(1,2*p2.shape[1],2),7] = -1*p2[0,:]*p1[1,:]
    HMat[range(0,2*p2.shape[1],2),8] = p2[1,:]
    HMat[range(1,2*p2.shape[1],2),8] = -1*p2[0,:]
    #print (HMat)
    U,S,Vh = np.linalg.svd(HMat)
    Vh = np.transpose(Vh)[:,-1]
    #print (Vh)
    H = np.reshape(Vh, (3,3))
    return H


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC
    
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    ###########################
    locs1 = (locs1[:,0:2])
    locs2 = (locs2[:,0:2])
    maxCount = 0
    for i in range(num_iter) :
        np.random.shuffle(matches)
        #HPointPairs = matches[0:4,:]
        p1 = np.empty([2,4])
        p2 = np.empty([2,4])
        for j in range(4) :
            p1[:,j] = np.transpose(locs1[matches[j][0],:])
            p2[:,j] = np.transpose(locs2[matches[j][1],:])
            #print (p1[:,j], p2[:,j])
        #print (p1.shape, p2.shape)
        HMat = computeH(p1, p2)
        #print (HMat)
        count = 0
        for k in range(matches.shape[0]) :
            inputMat = locs1[matches[k][0]]
            inputMat = np.append(inputMat,1)
            outputMat = np.matmul(HMat, inputMat)
            outputCart = np.array([outputMat[0]/outputMat[2], outputMat[1]/outputMat[2]])
            output = locs2[matches[k][1]]
            #print (outputCart)
            diff = np.sqrt(np.sum((output[0]-outputCart[0])**2 + (output[1]-outputCart[1])**2))
            #print (diff)
            if diff < tol :
                count += 1
        if (count > maxCount) :
            maxCount = count
            bestH = HMat 
    #print (bestH)
    print (maxCount)
    for loc in locs1 :
        loc = np.append(loc,1)
        A = np.matmul(bestH, loc)
        #print (A[0]/A[2], A[1]/A[2])
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    #cv2.imshow('1', im1)
    #cv2.imshow('2', im2)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc2, desc1)
    #p1 = np.array(np.transpose([[1,2],[4,5],[6,7],[8,9],[10,11]]))
    #p2 = np.array(np.transpose([[7,4],[3,6],[4,4],[2,5],[4,0]]))
    #H = computeH(p1, p2)
    H = ransacH(matches, locs2, locs1, num_iter=5000, tol=2)
    print (H)
    np.save('../results/q6_1.npy', H)
    #im3 = cv2.warpPerspective(im2, H, (2*im1.shape[1], 2*im1.shape[0]))
    #cv2.imshow('3',im3)
    #cv2.waitKey(0)