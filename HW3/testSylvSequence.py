import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
carSeq = np.load('../data/sylvseq.npy')
numFrames = carSeq.shape[2]
rectInit = np.array([101, 61, 155, 107])
bases = np.load('../data/sylvbases.npy')
rect = rectInit
rectVec = rect
rectBad = rectInit
printFrames = [ 1, 200, 300, 350, 400]
for i in range(numFrames - 1) :
	#print (i)
	topLeftY = rect[0]
	topLeftX = rect[1]
	bottomRightY = rect[2]
	bottomRightX = rect[3]
	pVec = LucasKanadeBasis(carSeq[:,:,i], carSeq[:,:,i + 1], rect, bases)
	topLeftYBad = rectBad[0]
	topLeftXBad = rectBad[1]
	bottomRightYBad = rectBad[2]
	bottomRightXBad = rectBad[3]
	pVecBad = LucasKanade(carSeq[:,:,i], carSeq[:,:,i + 1], rectBad)
	#print (pVec)
	if i in printFrames :
		fig,ax = plt.subplots(1)
		ax.imshow(carSeq[:,:,i], cmap = plt.cm.gray)
		rectShow = patches.Rectangle((topLeftY,topLeftX),bottomRightY - topLeftY,bottomRightX - topLeftX,linewidth=5,edgecolor='g',facecolor='none')
		rectShowBad = patches.Rectangle((topLeftYBad,topLeftXBad),bottomRightYBad - \
			topLeftYBad , bottomRightXBad - topLeftXBad, linewidth=5,edgecolor='y',facecolor='none')
		ax.add_patch(rectShowBad)
		ax.add_patch(rectShow)
		imName = str('sylv_' + str(i) + '.png')
		plt.savefig(imName)
		plt.close()
	rect = np.array([(topLeftY + pVec[1]), (topLeftX + pVec[0]), (bottomRightY + pVec[1]), (bottomRightX + pVec[0])])
	rectBad = np.array([(topLeftYBad + pVecBad[1]), (topLeftXBad + pVecBad[0]), \
		(bottomRightYBad + pVecBad[1]), (bottomRightXBad + pVecBad[0])])
	rectVec = np.vstack((rectVec, rect))

np.save('sylvseqrects.npy', rectVec)
