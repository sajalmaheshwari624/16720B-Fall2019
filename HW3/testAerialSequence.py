import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import scipy.ndimage

# write your script here, we recommend the above libraries for making your animation
carSeq = np.load('../data/aerialseq.npy')
numFrames = carSeq.shape[2]
motionMaskStack = np.zeros((carSeq.shape[0], carSeq.shape[1]))
count = 0
for i in range(1, numFrames) :
	#print (i)
	frame = np.zeros([carSeq[:,:,i-1].shape[0], carSeq[:,:,i].shape[1], 3])
	motionMask = SubtractDominantMotion(carSeq[:,:,i-1], carSeq[:,:,i])
	motionMask = scipy.ndimage.morphology.binary_erosion(motionMask, structure=np.eye(2))
	motionMask = scipy.ndimage.morphology.binary_dilation(motionMask, structure=np.ones((4,4)))
	frame[:,:,0] = carSeq[:,:,i]
	frame[:,:,1] = carSeq[:,:,i]
	frame[:,:,2] = carSeq[:,:,i] + motionMask * 0.8
	if i % 30 == 0 :
		fig,ax = plt.subplots(1)
		count += 1
		#print (count)
		ax.imshow(frame, cmap = plt.cm.gray)
		imName = str('LK_Affine' + str(i) + '.png')
		plt.savefig(imName)
		plt.close()
		if count == 1 :
			motionMaskStack = motionMask
		else :
			motionMaskStack = np.dstack((motionMaskStack, motionMask))
#print (motionMaskStack.shape)

np.save('aerialseqmasks.npy', motionMaskStack)

