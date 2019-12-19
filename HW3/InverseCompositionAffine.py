import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage
#import matplotlib.pyplot as plt
#import cv2
def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	numRows = It.shape[0]
	numCols = It.shape[1]
	templateIm = It
	currentIm = It1
	ImY, ImX = np.gradient(It)
	deltaPThresh = 0.01

	while 1 :
		MCopy = np.copy(M)
		MCopy[0:2, 0:2] = np.fliplr(MCopy[0:2, 0:2])
		MCopy = np.flipud(MCopy)
		buff = np.array([[0.0,0.0,1.0]])
		MCopy = np.append(MCopy, buff, axis = 0)
		currentImWarped = scipy.ndimage.affine_transform(currentIm, MCopy, cval = -500)
		validPoints = np.where(currentImWarped != -500)
		vectorB = currentImWarped[validPoints] - templateIm[validPoints]

		A = np.zeros([vectorB.size, 6])

		A[:,0] = ImX[validPoints] * validPoints[1]
		A[:,1] = ImX[validPoints] * validPoints[0]
		A[:,2] = ImX[validPoints]

		A[:,3] = ImY[validPoints] * validPoints[1]
		A[:,4] = ImY[validPoints] * validPoints[0]
		A[:,5] = ImY[validPoints]
		#print (A, vectorB)
		deltaP = np.linalg.lstsq(A, vectorB, rcond = -1)
		deltaP = deltaP[0]
		if np.linalg.norm(deltaP) < deltaPThresh :
			break
		else :
			deltaPCopy = np.copy(deltaP)
			deltaPCopy = np.reshape(deltaPCopy, (2,3))
			identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
			deltaPCopy += identity
			buffArr = np.array([[0.0,0.0,1.0]])
			deltaPCopy = np.append(deltaPCopy, buffArr, axis = 0)
			deltaPInv = np.linalg.inv(deltaPCopy)
			#print (MCopy)
			#print (deltaPCopy)
			MCopy2 = M
			MCopy2 = np.append(MCopy2, buffArr, axis = 0)
			newM = np.matmul(MCopy2, deltaPInv)
			#print (newM)
			M = newM[0:2, :]
			#print (M)

	return M


#aerSeq = np.load('../data/aerialseq.npy')
#M1 = InverseCompositionAffine(aerSeq[:,:,29], aerSeq[:,:,30])
#M2 = InverseCompositionAffine(aerSeq[:,:,30], aerSeq[:,:,29])
#print (M1, M2)
#buff = np.array([[0.0,0.0,1.0]])
#M1 = np.append(M1, buff, axis = 0)
#M2 = np.append(M2, buff, axis = 0)
#print ("M1", M1)
#print("M2", M2)
#print ("Product", np.matmul(M1, M2))
#print (M)