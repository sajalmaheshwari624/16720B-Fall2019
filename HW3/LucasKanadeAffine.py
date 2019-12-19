import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage
#import matplotlib.pyplot as plt
#import cv2
def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
	# put your implementation here
	M = np.array([[1., 0.0, 0.0], [0.0, 1.0, 0.0]])
	numRows = It.shape[0]
	numCols = It.shape[1]

	templateIm = It
	currentIm = It1
	ImY, ImX = np.gradient(It1)
	deltaPThresh = 0.001
	while 1 :
		#print (M)
		MCopy = np.copy(M)
		MCopy[0:2, 0:2] = np.fliplr(MCopy[0:2, 0:2])
		MCopy = np.flipud(MCopy)
		buff = np.array([[0.0,0.0,1.0]])
		MCopy = np.append(MCopy, buff, axis = 0)
		#print (MCopy)
		currentImWarped = scipy.ndimage.affine_transform(currentIm, MCopy, cval = -500)
		ImGridXWarped = scipy.ndimage.affine_transform(ImX, MCopy, cval = -500)
		ImGridYWarped = scipy.ndimage.affine_transform(ImY, MCopy, cval = -500)
		validPoints = np.where(currentImWarped != -500)
		#print (currentImWarped)
		#print (validPoints)
		#print (validPoints[0].shape)
		vectorB = It[validPoints] - currentImWarped[validPoints]
		#print(vectorB)
		A = np.zeros([vectorB.size, 6])

		A[:,0] = ImGridXWarped[validPoints] * validPoints[1]
		A[:,1] = ImGridXWarped[validPoints] * validPoints[0]
		A[:,2] = ImGridXWarped[validPoints]

		A[:,3] = ImGridYWarped[validPoints] * validPoints[1]
		A[:,4] = ImGridYWarped[validPoints] * validPoints[0]
		A[:,5] = ImGridYWarped[validPoints]

		#print (A)
		#print (vectorB)
		deltaP = np.linalg.lstsq(A, vectorB, rcond = -1)
		deltaP = deltaP[0]
		#print (deltaP, np.linalg.norm(deltaP))
		break
		if np.linalg.norm(deltaP) < deltaPThresh :
			break
		else :
			deltaPCopy = np.copy(deltaP)
			deltaPCopy = np.reshape(deltaP, M.shape)
			M += deltaPCopy
	return M

'''
aerSeq = np.load('../data/aerialseq.npy')
M1 = LucasKanadeAffine(aerSeq[:,:,0], aerSeq[:,:,1])
#M2 = LucasKanadeAffine(aerSeq[:,:,5], aerSeq[:,:,5])
buff = np.array([[0.0,0.0,1.0]])
M1 = np.append(M1, buff, axis = 0)
#M2 = np.append(M2, buff, axis = 0)
print ("M1", M1)
#print("M2", M2)
#print ("Product", np.matmul(M2, M1))
#print (M)
'''