import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
vectorMean = np.mean(train_x, axis = 0)
train_x_normalized = train_x - vectorMean
U,S,Vh = np.linalg.svd(train_x_normalized)
V = np.transpose(Vh)
PC_Vec = V[:,0:dim]
#print (PC_Vec.shape)
lowRankData = np.matmul(train_x, PC_Vec)
lrank = np.linalg.matrix_rank(PC_Vec)
#print (lrank) 

# rebuild it
PC_VecTr = np.transpose(PC_Vec)
recon = np.matmul(lowRankData, PC_VecTr) + vectorMean
# build valid dataset
recon_valid = None
vectorMeanValid = np.mean(valid_x, axis = 0)
valid_x_normalized = valid_x - vectorMeanValid
lowRankDataValid = np.matmul(valid_x_normalized, PC_Vec)
recon_valid = np.matmul(lowRankDataValid, PC_VecTr) + vectorMeanValid

indices = np.random.permutation(10)
indices = [14, 57, 489, 465, 1730, 1720, 2020, 2050, 3450, 3499]
for index in indices :
	xInput = valid_x[index]
	xInput = np.reshape(xInput, (32, 32))
	xInput = np.transpose(xInput)

	xOutput = recon_valid[index]
	xOutput = np.reshape(xOutput, (32, 32))
	xOutput = np.transpose(xOutput)

	plt.imshow(xInput)
	plt.show()

	plt.imshow(xOutput)
	plt.show()


valPSNR = 0
for i in range(valid_x.shape[0]) :
    real_out = valid_x[i]
    pred_out = recon_valid[i]
    valPSNR += psnr(real_out, pred_out)
print (valPSNR / valid_x.shape[0])
