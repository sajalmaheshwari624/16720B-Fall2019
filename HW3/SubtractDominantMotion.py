import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import scipy.ndimage
import matplotlib.pyplot as plt

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
	# put your implementation here
	mask = np.ones(image1.shape, dtype=bool)
	M =  LucasKanadeAffine(image1, image2)
	#M = InverseCompositionAffine(image1, image2)
	buff = np.array([[0.0,0.0,1.0]])
	M = np.append(M, buff, axis = 0)
	image2Estimated = scipy.ndimage.affine_transform(image1, M, np.shape(image1))
	diffImage = abs(image2 - image2Estimated)
	diffThresh = 0.2
	mask[diffImage < diffThresh] = 0
	return mask

#aerSeq = np.load('../data/aerialseq.npy')
#mask = SubtractDominantMotion(aerSeq[:,:,0], aerSeq[:,:,1])
#print (mask)
#plt.imshow(mask)
#plt.show()