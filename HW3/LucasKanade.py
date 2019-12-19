import numpy as np
from scipy.interpolate import RectBivariateSpline
#import matplotlib.pyplot as plt

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]

	# Put your implementation here
	topLeftY = int(np.round(rect[0]))
	topLeftX = int(np.round(rect[1]))
	bottomRightY = int(np.round(rect[2]))
	bottomRightX = int(np.round(rect[3]))

	currentTemp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
	tempGrid = np.meshgrid(np.linspace(topLeftX, bottomRightX, bottomRightX-topLeftX + 1),\
		np.linspace(topLeftY, bottomRightY, bottomRightY-topLeftY+1))
	interpolateTemp = currentTemp.ev(tempGrid[0], tempGrid[1])
	#print (interpolateTemp.shape)
	currentIm = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

	deltaPThresh = 0.001
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
		deltaP = np.linalg.lstsq(matrixA, vectorB, rcond = None)
		deltaP = deltaP[0]
		currentP[0] += deltaP[0]
		currentP[1] += deltaP[1]
		if np.linalg.norm(deltaP) < deltaPThresh :
			break
	return currentP

'''
carSeq = np.load('../data/carseq.npy')
It = carSeq[:,:,0]
It1 = carSeq[:,:,0]
rect = np.array([59, 116, 145, 151])
p0 = np.array([0.6, 0.5])
p = LucasKanade(It, It1, rect, p0)
print (p)
'''