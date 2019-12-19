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
rectVec = rect
for i in range(numFrames - 1) :
	#print (i)
	topLeftY = rect[0]
	topLeftX = rect[1]
	bottomRightY = rect[2]
	bottomRightX = rect[3]
	pVec = LucasKanade(carSeq[:,:,i], carSeq[:,:,i + 1], rect)
	#print (pVec)
	if i % 100 == 0 :
		fig,ax = plt.subplots(1)
		ax.imshow(carSeq[:,:,i], cmap = plt.cm.gray)
		rectShow = patches.Rectangle((topLeftY,topLeftX),bottomRightY - topLeftY,bottomRightX - topLeftX,linewidth=5,edgecolor='y',facecolor='none')
		ax.add_patch(rectShow)
		imName = str('frame' + str(i) + '.png')
		plt.savefig(imName)
		plt.close()
	rect = np.array([(topLeftY + pVec[1]), (topLeftX + pVec[0]), (bottomRightY + pVec[1]), (bottomRightX + pVec[0])])
	rectVec = np.vstack((rectVec, rect))

#print (rectVec)
np.save('carseqrects.npy', rectVec)