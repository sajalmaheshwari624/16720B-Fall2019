import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

#Testing only. REMOVE IN FINAL SUBMISSION
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = np.array([])
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    grayscaleIm = skimage.color.rgb2gray(image)
    grayscaleIm = skimage.filters.gaussian(grayscaleIm, sigma=1.0)
    threshVal= skimage.filters.threshold_otsu(grayscaleIm)
    threshIm = grayscaleIm <= threshVal
    threshClose = skimage.morphology.closing(threshIm, skimage.morphology.square(10))

    labelIm = skimage.measure.label(threshClose, connectivity = 1)
    
    count = 0
    for region in skimage.measure.regionprops(labelIm) :
    	count += 1
    	if count >= 1 :
    		if region.area >= 400 :
    			if bboxes.size == 0 :
    				bboxes = np.array(np.array(region.bbox))
    			else :
    				bboxes = np.vstack((bboxes, np.array(region.bbox)))

    bw =  (~(threshClose)).astype(np.float)
    return bboxes, bw

if __name__ == "__main__":
	#im = skimage.io.imread('../images/01_list.jpg')
	#bb, bw = findLetters(im)
	#im = skimage.io.imread('../images/02_letters.jpg')
	#bb, bw = findLetters(im)
	#im = skimage.io.imread('../images/03_haiku.jpg')
	#bb, bw = findLetters(im)
	#im = skimage.io.imread('../images/04_deep.jpg')
	#bb, bw = findLetters(im)
	#fig,ax = plt.subplots(1)
	#plt.imshow(bw)
	#plt.show()
	#print (bb)
	#ax.imshow(im)
	for i in range(bb.shape[0]) :
		topLeftY = bb[i,1]
		topLeftX = bb[i,0]
		bottomRightY = bb[i,3]
		bottomRightX = bb[i,2]
		rectShow = patches.Rectangle((topLeftY,topLeftX),bottomRightY - topLeftY,bottomRightX - topLeftX,linewidth=1,edgecolor='y',facecolor='none')
		ax.add_patch(rectShow)
	plt.show()
		