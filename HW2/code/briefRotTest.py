import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import BRIEF


im = cv2.imread('../data/model_chickenbroth.jpg')
dim = np.array([im.shape[1], im.shape[0]])
center = dim / 2
angle = np.array([])
numMatches = np.array([])
for i in range(0,361,10) :
	rotMat = cv2.getRotationMatrix2D(tuple(center), i, 1)
	rotIm = cv2.warpAffine(im, rotMat, tuple(dim))
	locs1, desc1 = BRIEF.briefLite(im)
	locs2, desc2 = BRIEF.briefLite(rotIm)
	matches = BRIEF.briefMatch(desc1, desc2)
	angle = np.append(angle, i)
	matchNum = matches.shape[0]
	#print (matchNum)
	numMatches = np.append(numMatches, matchNum)
	#cv2.imshow('a',rotIm)
	#cv2.waitKey(0)

plt.bar(angle, numMatches, alpha = 1)
#plt.xticks(angle)
plt.ylabel('Matches')
plt.savefig('../results/rotation_matches.png')
#plt.title('Brief number of matches v/s angle of rotation')

#plt.show()