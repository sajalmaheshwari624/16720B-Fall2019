import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #######################################
    # TO DO ...
    im3 = cv2.warpPerspective(im2, H2to1, (2*im1.shape[1], 2*im1.shape[0]))
    for i in range(im1.shape[0]) :
        for j in range(im1.shape[1]) :
            for k in range(im1.shape[2]) :
                op1 = im1[i,j,k]
                op2 = im3[i,j,k]
                if op2 == 0 :
                    im3[i,j,k] = op1

    return im3


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    ######################################
    
    pano_im = None
    im2TopRight = np.array([0, im2.shape[0]-1, 1])
    im2TransformedTopRight = np.matmul(H2to1, np.transpose(im2TopRight))
    im2TransformedTopRight = im2TransformedTopRight / im2TransformedTopRight[-1]
    im2TransformedTopRight = im2TransformedTopRight[0:2]
    #print (im2TransformedTopRight)
    im2TopLeft = np.array([0, 0, 1])
    im2TransformedTopLeft = np.matmul(H2to1, np.transpose(im2TopLeft))
    im2TransformedTopLeft = im2TransformedTopLeft / im2TransformedTopLeft[-1]
    im2TransformedTopLeft = im2TransformedTopLeft[0:2]
    #print (im2TransformedTopLeft)
    
    im2BottomLeft = np.array([im2.shape[1]-1, 0, 1])
    im2TransformedBottomLeft = np.matmul(H2to1, np.transpose(im2BottomLeft))
    im2TransformedBottomLeft = im2TransformedBottomLeft / im2TransformedBottomLeft[-1]
    im2TransformedBottomLeft = im2TransformedBottomLeft[0:2]
    #print (im2TransformedBottomLeft)

    im2BottomRight = np.array([im2.shape[1]-1, im2.shape[0]-1,1])
    im2TransformedBottomRight = np.matmul(H2to1, np.transpose(im2BottomRight))
    im2TransformedBottomRight = im2TransformedBottomRight / im2TransformedBottomRight[-1]
    im2TransformedBottomRight = im2TransformedBottomRight[0:2]
    #print (im2TransformedBottomRight)

    numCols = int(im2TransformedBottomLeft[0])
    numRows = int(im2TransformedBottomRight[1] - im2TransformedBottomLeft[1])
    #print (numCols, numRows)
    M = np.array([[1,0,0],[0,1, -1*im2TransformedBottomLeft[1]],[0,0,1]])
    im1toim3 = cv2.warpPerspective(im1, M, (int(numCols), int(numRows)))
    im2toim3 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (int(numCols), int(numRows)))
    #cv2.imshow('1to3', im1toim3)
    #cv2.waitKey(0)
    #cv2.imshow('2to3', im2toim3)
    #cv2.waitKey(0)
    im3 = np.zeros([numRows, numCols, 3])
    for i in range(numRows) :
        for j in range(numCols) :
            for k in range(3) :
                op1 = im1toim3[i,j,k]
                op2 = im2toim3[i,j,k]
                if (op1 == 0 and op2 != 0) :
                    im3[i,j,k] = op2
                elif (op2 == 0 and op1 != 0) :
                    im3[i,j,k] = op1
                else :
                    im3[i,j,k] = op1
    return im3

def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc2, desc1)
    H = np.load('../results/q6_1.npy')
    #H = ransacH(matches, locs2, locs1, num_iter=5000, tol=2)
    im3 = imageStitching_noClip(im1, im2, H)
    cv2.imwrite('../results/panorama.png', im3)
    #cv2.waitKey(0)
    

if __name__ == '__main__':

    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc2, desc1)
    
    #H = ransacH(matches, locs2, locs1, num_iter=5000, tol=2)
    #np.save('../results/q6_1.npy', H)
    #H = np.load('H2to1.npy')
    #cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    #im3 = imageStitching_noClip(im1, im2, H)
    #im3 = cv2.warpPerspective(im2, H, (im2.shape[1], im2.shape[0]))
    #cv2.imshow('Pyramid of image', im3)
    #cv2.waitKey(0)
    generatePanaroma(im1, im2)