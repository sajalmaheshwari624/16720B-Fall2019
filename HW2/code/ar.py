import numpy as np
import cv2
import os
from planarH import computeH
import matplotlib.pyplot as plt

def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...
    H1 = np.matmul(np.linalg.inv(K),H)
    #print (H1)
    H2Col = H1[:,0:2]
    HTrans = H1[:,2]
    U,S,Vh = np.linalg.svd(H2Col)
    oneMat = np.array([[1,0],[0,1],[0,0]])
    R2Col = np.matmul(U, np.matmul(oneMat, Vh))
    RCol3 = np.cross(R2Col[:,0], R2Col[:,1])
    #print (RCol3)
    RMat = np.zeros([3,3])
    RMat[:,0] = R2Col[:,0]
    RMat[:,1] = R2Col[:,1]
    RMat[:,2] = RCol3
    normR2Col = H2Col / R2Col
    lambaprime = np.sum(normR2Col) / 6
    #print (lambaprime)
    t = HTrans / lambaprime
    #print (np.linalg.det(RMat))
    RMat = RMat
    #print (RMat, t)

    return RMat, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...
    #print (W.shape, R.shape, K.shape)
    ExMat = np.zeros([3,4])
    ExMat[:,0:3] = R
    ExMat[:,3] = t
    #print (ExMat)
    x = np.matmul(np.matmul(K, ExMat), W)
    X = (x[0:2, :] / x[2, :]).astype(int)
    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')

    #############################
    # TO DO ...
    W = np.array([[0.0, 18.2, 18.2, 0.0],[0.0, 0.0, 26.0, 26.0],[0.0, 0.0, 0.0, 0.0]])
    X = np.array([[483, 1704, 2175, 67],[810, 781,  2217, 2286]])
    K = np.array([[3043.72, 0.0, 1196.0],[0.0, 3043.72, 1604.0],[0.0, 0.0, 1.0]])

    with open('../data/sphere.txt', "r") as inFile:
        inData = inFile.readlines()

    inDataX = inData[0].split('  ')
    inDataY = inData[1].split('  ')
    inDataZ = inData[2].split('  ')
    #print (len(inDataX), len(inDataY), len(inDataZ))

    sphere3DPoints = np.array([[],[],[]])
    for i in range(1, len(inDataX)):
        sphere3DPoints = np.append(sphere3DPoints, np.array([[float(inDataX[i]) + 11], [float(inDataY[i]) + 16], [float(inDataZ[i])]]), axis = 1)
    sphere3DPoints = np.append(sphere3DPoints, np.ones((1, sphere3DPoints.shape[1])), axis = 0)
    #print (sphere3DPoints)

    W_2D = W[0:2, :]
    #print (X)
    H = computeH(W_2D, X)
    #print (H)
    R, t = compute_extrinsics(K, H)
    np.transpose(t)
    #print (R, t)
    imagePoints = project_extrinsics(K, sphere3DPoints, R, t)
    fig = plt.figure()
    plt.imshow(im)

    for i in range(imagePoints.shape[1]):
        plt.plot(imagePoints[0, i], imagePoints[1, i], 'y.', markersize = 1)
    #plt.draw()
    #plt.waitforbuttonpress(0)
    #plt.close(fig)
    plt.savefig('../results/arImage.png')
    #plt.show()