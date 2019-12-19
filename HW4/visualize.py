import numpy as np
import submission
import helper
import findM2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
pts = np.load('../data/templeCoords.npz')
xPoints = pts['x1']
yPoints = pts['y1']

templePoints = np.append(xPoints, yPoints, axis = 1)
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(im1.shape)

pts = np.load('../data/some_corresp.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']

intrinsics = np.load('../data/intrinsics.npz')
FEight = submission.eightpoint(pts1, pts2, M)
M2,C2,_ = findM2.test_M2_solution(pts1, pts2, intrinsics, M)
#print (M2, C2)

M1 = np.zeros([3,4])
M1[0,0] = 1
M1[1,1] = 1
M1[2,2] = 1
K1 = intrinsics['K1']
C1 = np.matmul(K1, M1)
im2Array = np.array([])
#helper.epipolarMatchNoGUI(im1, im2, FEight, xPoints, yPoints)

for i in range(xPoints.shape[0]) :
	x = int(xPoints[i])
	y = int(yPoints[i])
	correspPoint = submission.epipolarCorrespondence(im1, im2, FEight, x, y)
	#print (correspPoint)
	#correspPoint = np.reshape(correspPoint, (1,2))
	if i == 0 :
		im2Array = correspPoint
	else :
		im2Array = np.vstack((im2Array, correspPoint))

Points3D, error = submission.triangulate(C1, templePoints, C2, im2Array)
#print (error)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xmin, xmax = np.min(Points3D[:, 0]), np.max(Points3D[:, 0])
ymin, ymax = np.min(Points3D[:, 1]), np.max(Points3D[:, 1])
zmin, zmax = np.min(Points3D[:, 2]), np.max(Points3D[:, 2])

ax.set_xlim3d(xmin, xmax)
ax.set_xlabel('x')
ax.set_ylim3d(ymin, ymax)
ax.set_ylabel('y')
ax.set_zlim3d(zmin, zmax)
ax.set_zlabel('z')

ax.scatter(Points3D[:, 0], Points3D[:, 1], Points3D[:, 2], c='b', marker='o')
plt.show()

outFile = 'q4_2.npz'
np.savez(outFile, F = FEight, M1 = M1, M2 = M2, C1 = C1, C2 = C2)