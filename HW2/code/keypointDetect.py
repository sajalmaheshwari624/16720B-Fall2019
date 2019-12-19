import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''
    
    DoG_pyramid = np.zeros((gaussian_pyramid.shape[0], gaussian_pyramid.shape[1]))
    ################
    # TO DO ...
    # compute DoG_pyramid here
    for i in range(len(levels)-1) :
        if i == 0 :
            DoG_pyramid = gaussian_pyramid[:,:,i+1] - gaussian_pyramid[:,:,i]
        else :
            #outputIm = np.stack((outputredChannel, green, outputBlueChannel), axis=-1)
            DoG_pyramid = np.dstack((DoG_pyramid, gaussian_pyramid[:,:,i+1] - gaussian_pyramid[:,:,i]))
    
    DoG_levels = levels[1:]
    #print("DoG sum:", np.sum(DoG_pyramid))
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = np.zeros((DoG_pyramid.shape))
    
    ##################
    # TO DO ...
    # Compute principal curvature here
    for i in range(principal_curvature.shape[2]) :
        currentImage = DoG_pyramid[:,:,i]
        Ixx = cv2.Sobel(cv2.Sobel(currentImage,cv2.CV_32F,1,0,ksize = -1), cv2.CV_32F,1,0, ksize = -1)
        Ixy = cv2.Sobel(cv2.Sobel(currentImage,cv2.CV_32F,0,1,ksize = -1), cv2.CV_32F,1,0, ksize = -1)
        Iyx = cv2.Sobel(cv2.Sobel(currentImage,cv2.CV_32F,1,0,ksize = -1), cv2.CV_32F,0,1, ksize = -1)
        Iyy = cv2.Sobel(cv2.Sobel(currentImage,cv2.CV_32F,0,1,ksize = -1), cv2.CV_32F,0,1, ksize = -1)
        detH = np.multiply(Ixx, Iyy) - np.multiply(Iyx, Ixy) + pow(10,-8)
        trH = np.multiply((Ixx + Iyy), (Ixx + Iyy))
        RCurrentImage = np.divide(trH, detH)
        principal_curvature[:,:,i] = RCurrentImage
        #print (RCurrentImage)
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    
    ##############
    #  TO DO ...
    # Compute locsDoG here
                
    count = 0
    for i in range(1,DoG_pyramid.shape[0]-1) :
        for j in range(1,DoG_pyramid.shape[1]-1) :
            diffLevels = DoG_levels
            for k in diffLevels :
                imThisLevel = DoG_pyramid[i,j,k]
                if k == 0 :
                    diffBelow = 1
                    imAboveLevel = DoG_pyramid[i,j,k+1]
                    diffAbove = imThisLevel - imAboveLevel
                elif k == 4 :
                    diffAbove = 1
                    imBelowLevel = imBelowLevel = DoG_pyramid[i,j,k-1]
                    diffBelow = imThisLevel - imBelowLevel
                else :
                    imAboveLevel = DoG_pyramid[i,j,k+1]
                    imBelowLevel = DoG_pyramid[i,j,k-1]
                    diffAbove = imThisLevel - imAboveLevel
                    diffBelow = imThisLevel - imBelowLevel
                imCurvature = principal_curvature[i,j,k]
                if (diffAbove * diffBelow > 0 and imCurvature < th_r and imCurvature > 0) :
                    imStack = DoG_pyramid[i-1:i+2, j-1:j+2, k]
                    maxValue = np.max(imStack)
                    #print (maxValue)
                    maxIndex = np.argmax(imStack)
                    minValue = np.min(imStack)
                    minIndex = np.argmin(imStack)
                    #print (maxValue, minValue, maxIndex, minIndex)
                    if (maxIndex == 4 or minIndex == 4) :
                        if (abs(imStack[1,1]) >= th_contrast) :
                            count += 1
                            if count == 1 :
                                locsDoG = np.array([j,i,k])
                            else :
                                locsDoG = np.vstack((locsDoG, np.array([j,i,k])))
    return locsDoG
  

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here
    im_pyr = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    '''
    print (locsDoG.shape)
    #print (locsDoG.shape)
    color = (0,255,0)
    radius = 1
    thickness = 1
    for i in range(locsDoG.shape[0]) :
        im = cv2.circle(im, tuple((locsDoG[i,0:2])), radius, color, thickness)

    im = cv2.resize(im, (5*im.shape[1], 5*im.shape[0]))
    cv2.imshow('Pyramid of image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return locsDoG, im_pyr


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    #locsDoG, gaussian_pyramid = DoGdetector(im)
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locsDoG, gaussian_pyramid = DoGdetector(im)
    #displayPyramid(gaussian_pyramid)