import numpy as np
import submission
import helper
import matplotlib.pyplot as plt

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def test_M2_solution(pts1, pts2, intrinsics, M):
	'''
	Estimate all possible M2 and return the correct M2 and 3D points P
	:param pred_pts1:
	:param pred_pts2:
	:param intrinsics:
	:param M: a scalar parameter computed as max (imwidth, imheight)
	:return: M2, the extrinsics of camera 2
			 C2, the 3x4 camera matrix
			 P, 3D points after triangulation (Nx3)
	'''
	FEight = submission.eightpoint(pts1, pts2, M)
	KIntrinsic1 = intrinsics['K1']
	KIntrinsic2 = intrinsics['K2']
	E = submission.essentialMatrix(FEight, KIntrinsic1, KIntrinsic2)
	ExtrinsicMList = helper.camera2(E)
	ProjMatrix1 = np.zeros([3,4])
	ProjMatrix1[0,0] = 1
	ProjMatrix1[1,1] = 1
	ProjMatrix1[2,2] = 1

	maxCount = -1
	minError = 999999999
	bestC2 = np.zeros([3,4])
	P = np.zeros((pts1.shape[0], 3))
	for i in range(ExtrinsicMList.shape[2]) :
		M1 = ProjMatrix1
		M2 = ExtrinsicMList[:,:,i]
		[W1, err1] = submission.triangulate(np.matmul(KIntrinsic1,M1), pts1, np.matmul(KIntrinsic2,M2), pts2)
		zIndicesCam1 = W1[:,2]
		validZValCam1 = np.where(zIndicesCam1 > 0)

		Rinv = np.linalg.inv(M2[:,0:3])
		tInv = -np.matmul(np.linalg.inv(M2[:,0:3]), M2[:,3])

		M2_new = ProjMatrix1
		M1_new = np.zeros((3,4))
		M1_new[:,0:3] = Rinv
		M1_new[:,3] = tInv

		[W2, err2] = submission.triangulate(np.matmul(KIntrinsic1, M1_new), pts1, np.matmul(KIntrinsic2, M2_new), pts2)
		zIndicesCam2 = W2[:,2]
		validZValCam2 = np.where(zIndicesCam2 > 0)
		#print (validZValCam2, validZValCam1)
		validZBothCam = np.intersect1d(validZValCam1, validZValCam2)
		#print (validZBothCam)		
		validCount = validZBothCam.size
		if validCount > maxCount :
			maxCount = validCount
			maxError = err1
			bestM2 = M2
			P = W1

	#print (maxCount)
	M2 = bestM2
	C2 = np.matmul(KIntrinsic2,M2)
	#print (P)
	return M2, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	intrinsics = np.load('../data/intrinsics.npz')
	pts1Max = np.max(pts1[:])
	pts2Max = np.max(pts2[:])
	im1 = plt.imread('../data/im1.png')
	im2 = plt.imread('../data/im2.png')
	M = max(im1.shape)
	#print (M)
	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics, M)
	print (M2, C2)
	#print (M2)
	np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
