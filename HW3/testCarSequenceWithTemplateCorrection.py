import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
carSeq = np.load('../data/carseq.npy')
numFrames = carSeq.shape[2]
rectInit = np.array([59, 116, 145, 151])
rect = rectInit
badRect = rectInit
rectVec = rect
pVecPrev = np.zeros(2)
firstFrame = carSeq[:,:,0]
epsilon = 4
for i in range(numFrames - 1) :
	#print (i)
	topLeftY = rect[0]
	topLeftX = rect[1]
	bottomRightY = rect[2]
	bottomRightX = rect[3]

	topLeftYBad = badRect[0]
	topLeftXBad = badRect[1]
	bottomRightYBad = badRect[2]
	bottomRightXBad = badRect[3]
	
	pVec = LucasKanade(carSeq[:,:,i], carSeq[:,:,i + 1], rect)
	initGuess = np.array([rect[1] - rectInit[1], rect[0] - rectInit[0]])
	#print (initGuess)
	pVecInit = LucasKanade(firstFrame, carSeq[:,:,i+1], rectInit, pVec + initGuess)
	pDiffInit = pVecInit - initGuess
	#print (pVecInit - initGuess)
	pDiff = pVec - pDiffInit
	#print (pDiff)
	if np.linalg.norm(pDiff) < epsilon :
		pVecPrev = pVec
		updateVec = pVec
	else :
		updateVec = pVecInit - initGuess
	#print (updateVec)
	if i % 100 == 0 :
		#fig,ax = plt.subplots(1)
		fig,ax = plt.subplots(1)
		ax.imshow(carSeq[:,:,i], cmap = plt.cm.gray)
		rectShow = patches.Rectangle((topLeftY,topLeftX),bottomRightY - topLeftY,bottomRightX - topLeftX,linewidth=5,edgecolor='y',facecolor='none')
		ax.add_patch(rectShow)
		rectShowBad = patches.Rectangle((topLeftYBad,topLeftXBad),bottomRightYBad - \
			topLeftYBad,bottomRightXBad - topLeftXBad,linewidth=5,edgecolor='g',facecolor='none')
		ax.add_patch(rectShowBad) 
		imName = str('frame_corrected' + str(i) + '.png')
		plt.savefig(imName)
		plt.close()
	rect = np.array([(topLeftY + updateVec[1]), (topLeftX + updateVec[0]), (bottomRightY + updateVec[1]), (bottomRightX + updateVec[0])])
	badRect = np.array([(topLeftYBad + pVec[1]), (topLeftXBad + pVec[0]), \
		(bottomRightYBad + pVec[1]), (bottomRightXBad + pVec[0])])
	rectVec = np.vstack((rectVec, rect))

#print (rectVec)
np.save('carseqrects-wcrt.npy', rectVec)
	
