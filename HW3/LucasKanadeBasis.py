import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]
	topLeftY = int(np.round(rect[0]))
	topLeftX = int(np.round(rect[1]))
	bottomRightY = int(np.round(rect[2]))
	bottomRightX = int(np.round(rect[3]))
	p0 = np.zeros(2)
	currentTemp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
	tempGrid = np.meshgrid(np.linspace(topLeftX, bottomRightX, bottomRightX-topLeftX + 1),\
		np.linspace(topLeftY, bottomRightY, bottomRightY-topLeftY+1))
	interpolateTemp = currentTemp.ev(tempGrid[0], tempGrid[1])

	currentIm = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
	#print (bases.shape)
	basesVectors = np.reshape(bases, [bases.shape[0] * bases.shape[1], bases.shape[2]])
	basesVectorsTr = np.transpose(basesVectors)
	bbTranspose = np.matmul(basesVectors, basesVectorsTr)
	#print (basesVectors.shape)
	#return None
	deltaPThresh = 0.01
	currentP = p0
	while 1 :
		currentImGrid = np.meshgrid(np.linspace(topLeftX + currentP[0], bottomRightX + currentP[0], bottomRightX+1 - topLeftX), \
			np.linspace(topLeftY + currentP[1], bottomRightY + currentP[1], bottomRightY+1 - topLeftY))

		interpolateIm = currentIm.ev(currentImGrid[0], currentImGrid[1])
		imX = currentIm.ev(currentImGrid[0], currentImGrid[1], dx = 1)
		imY = currentIm.ev(currentImGrid[0], currentImGrid[1], dy = 1)

		matrixB = interpolateTemp - interpolateIm
		vectorB = np.reshape(matrixB, [matrixB.size, 1])
		vectorAx = np.reshape(imX, [imX.size, 1])
		vectorAy = np.reshape(imY, [imY.size, 1])
		matrixA = np.hstack((vectorAx, vectorAy))
		matrixABBT = np.matmul(bbTranspose,matrixA)
		vectorBBT = np.matmul(bbTranspose, vectorB)
		#print (matrixA.shape, vectorB.shape)
		#print (matrixABBT.shape, vectorBBT.shape)
		finalMatrixA = matrixA - matrixABBT
		finalVectorB = vectorB - vectorBBT
		deltaP = np.linalg.lstsq(finalMatrixA, finalVectorB, rcond = None)
		deltaP = deltaP[0]
		currentP[0] += deltaP[0]
		currentP[1] += deltaP[1]
		if np.linalg.norm(deltaP) < deltaPThresh :
			break
	return currentP
