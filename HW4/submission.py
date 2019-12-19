"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import matplotlib.pyplot as plt
import scipy.optimize
import findM2
from mpl_toolkits.mplot3d import Axes3D
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    normPts1 = pts1/M
    normPts2 = pts2/M
    
    pts1Y = normPts1[:,1]
    pts1X = normPts1[:,0]

    pts2Y = normPts2[:,1]
    pts2X = normPts2[:,0]

    AMat = np.ones([np.size(pts1X),9])

    AMat[:,0] = np.multiply(pts2X, pts1X)
    AMat[:,1] = np.multiply(pts2X, pts1Y)
    AMat[:,2] = pts2X
    AMat[:,3] = np.multiply(pts2Y, pts1X)
    AMat[:,4] = np.multiply(pts2Y, pts1Y)
    AMat[:,5] = pts2Y
    AMat[:,6] = pts1X
    AMat[:,7] = pts1Y

    U,S,Vt = np.linalg.svd(AMat)
    VNormal = np.transpose(Vt)
    Fiter1 = VNormal[:,-1]
    Fiter1Res = np.reshape(Fiter1, (3,3))
    Unew, Snew, Vtnew = np.linalg.svd(Fiter1Res)
    Snew[-1] = 0
    SMat = np.diag(Snew)
    Fiter2 = np.matmul(np.matmul(Unew, SMat),Vtnew)
    #print (Fiter2)
    Fscaled = helper.refineF(Fiter2, normPts1, normPts2)
    TMat = np.array([[1/M, 0, 0],[0, 1/M, 0], [0,0,1]])
    Funscaled = np.matmul(np.matmul(TMat, Fscaled), TMat)
    return Funscaled


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    normPts1 = pts1/M
    normPts2 = pts2/M
    
    pts1Y = normPts1[:,1]
    pts1X = normPts1[:,0]

    pts2Y = normPts2[:,1]
    pts2X = normPts2[:,0]

    AMat = np.ones([np.size(pts1X),9])

    AMat[:,0] = np.multiply(pts2X, pts1X)
    AMat[:,1] = np.multiply(pts2X, pts1Y)
    AMat[:,2] = pts2X
    AMat[:,3] = np.multiply(pts2Y, pts1X)
    AMat[:,4] = np.multiply(pts2Y, pts1Y)
    AMat[:,5] = pts2Y
    AMat[:,6] = pts1X
    AMat[:,7] = pts1Y

    #print (AMat)

    U,S,Vt = np.linalg.svd(AMat)
    VNormal = np.transpose(Vt)

    Fiter1 = VNormal[:,-1]
    Fiter1Res = np.reshape(Fiter1, (3,3))
   
    Fiter2 = VNormal[:,-2]
    Fiter2Res = np.reshape(Fiter2, (3,3))

    #print (Fiter1, Fiter2)

    alphaFunc = lambda alpha: np.linalg.det(alpha * Fiter1Res + (1 - alpha) * Fiter2Res)

    a0=alphaFunc(0)
    a1 = 2 * (alphaFunc(1) - alphaFunc(-1))/3 - (alphaFunc(2) - alphaFunc(-2))/12
    a2=0.5*alphaFunc(1) + 0.5*alphaFunc(-1) - alphaFunc(0)
    a3 = (alphaFunc(1) - alphaFunc(-1) - 2*a1)/2
    coeffArray = np.array([a3, a2, a1, a0])
    alphaValues = np.roots(coeffArray)
    #print (alphaValues)
    realAlphaIndices = np.isreal(alphaValues)
    realAlphaVal = alphaValues[realAlphaIndices == True]
    #print (realAlphaVal)
    Farray = []
    for i in range(len(realAlphaVal)) :
        FMatrix = np.real(realAlphaVal[i])*Fiter1 + (1 - np.real(realAlphaVal[i]))*Fiter2
        FMatrix = np.reshape(FMatrix, (3,3))
        #Fscaled = helper.refineF(FMatrix, normPts1, normPts2)
        #print (Fscaled)
        TMat = np.array([[1/M, 0, 0],[0, 1/M, 0], [0,0,1]])
        Funscaled = np.matmul(np.matmul(TMat, FMatrix), TMat)
        Farray.append(Funscaled)

    return Farray
'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return np.matmul(np.matmul(np.transpose(K2), F), K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    #print ("Mandir wahin banaenge!")
    pts1X = pts1[:,0]
    pts1Y = pts1[:,1]

    pts2X = pts2[:,0]
    pts2Y = pts2[:,1]

    points3D = np.zeros((pts1X.size, 3))
    for index in range(pts1X.size) :
        AMat = np.zeros((4,4))
        AMat[0,:] = pts1X[index] * C1[2,:] - C1[0,:]
        AMat[1,:] = pts1Y[index] * C1[2,:] - C1[1,:]
        AMat[2,:] = pts2X[index] * C2[2,:] - C2[0,:]
        AMat[3,:] = pts2Y[index] * C2[2,:] - C2[1,:]
        U,S,Vt = np.linalg.svd(AMat)

        V = np.transpose(Vt)
        point4D = V[:,-1]
        point4D = point4D / point4D[-1]
        point3D = point4D[0:3]
        points3D[index,:] = point3D

    #print (points3D)

    error = 0
    for index in range(pts1X.size) :
        point3D = points3D[index,:]
        #print (point3D)
        point4D = np.append(point3D, np.array([1]))
        #print (point4D)
        projPointCam1 = np.matmul(C1, point4D)
        projPointCam2 = np.matmul(C2, point4D)
        projPointCam1_2D = projPointCam1 / projPointCam1[-1]
        projPointCam2_2D = projPointCam2 / projPointCam2[-1]
        projPointCam1_2D = projPointCam1_2D[0:-1]
        projPointCam2_2D = projPointCam2_2D[0:-1]
        givenPointCam1 = pts1[index]
        givenPointCam2 = pts2[index]
        #if index == 100 :
            #print (point4D)
            #print (projPointCam1_2D, projPointCam2_2D)
        norm1 = np.sum(np.square((givenPointCam1 - projPointCam1_2D)))
        norm2 = np.sum(np.square((givenPointCam2 - projPointCam2_2D)))
        error += norm1 + norm2
    return points3D, error

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''

def gaussianImage(imPatch, winSize) :
    x = np.arange(-winSize, winSize)
    y = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2.)
    y = np.reshape(y, [y.size,1])
    Y2D = np.matmul(y, np.transpose(y))
    gaussOut1 = np.multiply(Y2D, imPatch[:,:,0])
    gaussOut2 = np.multiply(Y2D, imPatch[:,:,1])
    gaussOut3 = np.multiply(Y2D, imPatch[:,:,2])
    gaussOut = np.stack((gaussOut1, gaussOut2, gaussOut3), axis=-1)
    return gaussOut

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation

    v = np.array([x1,y1,1])
    #print (v)
    windowSize = 5
    gaussianSigma = 1
    #print (y1 - windowSize, y1 + windowSize, x1 - windowSize, x1 + windowSize)
    im1Patch = im1[y1 - windowSize : y1 + windowSize, x1 - windowSize : x1 + windowSize, :]
    im1PatchGauss = gaussianImage(im1Patch,windowSize)
    l = np.dot(np.transpose(v),np.transpose(F))
    closeLimit = 40
    possiblePoints = np.array([])
    for i in range(windowSize, im2.shape[0] - windowSize) :
        yCoord = i
        xCoord = (-l[2] - l[1]*yCoord) / l[0]
        dist = np.sqrt(np.sum(np.square(x1 - xCoord) + np.square(y1 - yCoord)))
        if dist < closeLimit :
            validPoint = np.array([xCoord, yCoord])
            if possiblePoints.size == 0 :
                possiblePoints = validPoint
            else :
                possiblePoints = np.vstack((possiblePoints, validPoint))

    #print (possiblePoints)
    minError = 100000000
    matchPoint = np.zeros((2,1))
    for points in possiblePoints :
        #print (points)
        im2Patch = im2[int(points[1]) - windowSize: int(points[1]) + windowSize, int(points[0]) - windowSize : int(points[0]) + windowSize, :]
        im2PathGauss = gaussianImage(im2Patch , windowSize)
        error = np.sum(abs(im1PatchGauss - im2PathGauss))
        if error < minError :
            matchPoint = points
            minError = error
    #print (matchPoint, minError, x1, y1)
    return matchPoint

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    numPoints = pts1.size // 2
    #print (numPoints)
    thresh = 0.001
    maxIter = 5000
    finalInliers = 0
    finalF = np.zeros((3,3))
    finalBoolVec = np.zeros(numPoints, dtype = bool)
    for iters in range(maxIter) :
        #print (iters)
        randPerm = np.random.permutation(numPoints)
        randPoints = randPerm[0:7]
        F = sevenpoint(pts1[randPoints,:], pts2[randPoints,:], M)
        #print (F)
        for FMatrix in F :
            boolVec = np.zeros(numPoints, dtype = bool)
            for i in range(numPoints) :
                point1 = np.append(pts1[i,:], 1)
                point2 = np.append(pts2[i,:], 1)
                val = abs(np.matmul(np.matmul(np.transpose(point2), FMatrix),point1))
                #print (val)
                if val < thresh :
                    boolVec[i] = True
            numInliers = boolVec[boolVec == True].size
            if numInliers > finalInliers :
                finalInliers = numInliers
                finalF = FMatrix
                finalBoolVec = boolVec
    pts1Valid = pts1[np.where(finalBoolVec == True)]
    pts2Valid = pts2[np.where(finalBoolVec == True)]
    FEight = eightpoint(pts1Valid, pts2Valid, M)
    print (pts1Valid.shape)
    #print (FEight)
    return FEight, finalBoolVec



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    I = np.eye(3)
    R = np.zeros((3,3))
    theta = np.linalg.norm(r)
    u = r / theta
    if theta == 0 :
        R = I
    else :
        uTr = np.transpose(u)
        Ux = np.zeros((3,3))
        Ux[0,1] = -u[2]
        Ux[1,0] = u[2]
        Ux[0,2] = u[1]
        Ux[2,0] = -u[1]
        Ux[1,2] = -u[0]
        Ux[2,1] = u[0]
        R = I*np.cos(theta) + (1 - np.cos(theta))*(np.matmul(u, uTr)) + np.sin(theta)*Ux
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    #print (theta)
    epsilon = 0.0000001
    if theta < epsilon :
        return np.zeros((3,1))   
    r = np.array([[R[2,1] - R[1,2]], [R[0,2] - R[2,0]], [R[1,0] - R[0,1]]])
    r = 1.0 / (2 * np.sin(theta)) * r
    rFinal = theta * r
    #print (rFinal.shape)
    return rFinal

    # Replace pass by your implementation


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    numPoints = x.size // 3 - 2
    #print (numPoints)
    points3D = x[0 : numPoints*3]
    points3D = np.reshape(points3D, (points3D.size // 3, 3))
    rVec = x[numPoints*3: numPoints*3 + 3]
    rVec = np.reshape(rVec, (rVec.size,1))
    #print (rVec)
    t = x[numPoints*3 + 3:]
    #print (x.shape)

    RMat = rodrigues(rVec)
    #print (RMat)
    t = np.reshape(t, (3,1))
    M2 = np.append(RMat, t, axis = 1)
    C1 = np.matmul(K1,M1)
    C2 = np.matmul(K2, M2)
    #print (RMat, t)
    error = np.array([])

    for index in range(0, numPoints) :
        point3D = points3D[index,:]
        #print (point3D)
        point4D = np.append(point3D, np.array([1]))
        projPointCam1 = np.matmul(C1, point4D)
        projPointCam2 = np.matmul(C2, point4D)
        projPointCam1_2D = projPointCam1 / projPointCam1[-1]
        projPointCam2_2D = projPointCam2 / projPointCam2[-1]
        projPointCam1_2D = projPointCam1_2D[0:-1]
        projPointCam2_2D = projPointCam2_2D[0:-1]
        givenPointCam1 = p1[index]
        givenPointCam2 = p2[index]
        error1 = givenPointCam1 - projPointCam1_2D
        error2 = givenPointCam2 - projPointCam2_2D
        error1 = np.reshape(error1, (error1.size,1))
        error2 = np.reshape(error2, (error2.size,1))
        errorBoth = np.append(error1, error2, axis = 0)
        #print (errorBoth)
        if error.size == 0 :
            error = errorBoth
        else :
            error = np.append(error, errorBoth, axis = 0)
    #print (error.shape)
    return error 



'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    numPoints = P_init.shape[0]
    #print (numPoints)
    residualError = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x).flatten()

    initXVec = np.zeros((3*numPoints + 6), dtype = float)
    p_init_flatten = np.reshape(P_init, (P_init.size))
    initXVec[0:-6] = p_init_flatten
    RMat = M2_init[:,0:3]
    rVec = invRodrigues(RMat)
    rVec = np.reshape(rVec, (rVec.size))
    tVec = M2_init[:,3]
    initXVec[-6:-3] = rVec
    initXVec[-3:] = tVec

    #print (np.square(np.linalg.norm(residualError(initXVec))))
    xFinal,_ = scipy.optimize.leastsq(residualError, initXVec)
    #print (xFinal)
    #print (np.square(np.linalg.norm(residualError(xFinal))))
    wFinal = xFinal[0 : numPoints*3]
    rFinal = xFinal[numPoints*3 : numPoints*3+3]
    tFinal = xFinal[numPoints*3+3:]
    points3D = xFinal[0 : numPoints*3]
    points3D = np.reshape(points3D, (points3D.size // 3, 3))
    rVec = xFinal[numPoints*3: numPoints*3 + 3]
    rVec = np.reshape(rVec, (rVec.size,1))
    t = xFinal[numPoints*3 + 3:]


    M2Final = np.zeros((3,4))
    M2Final[:,0:3] = rodrigues(rFinal)
    M2Final[:,3] = tFinal
    return M2Final, points3D



if __name__ == "__main__" :
    pts = np.load('../data/some_corresp_noisy.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = max(im1.shape)
    pts1 = pts['pts1']
    pts2 = pts['pts2']

    FRansac, inliers = ransacF(pts1, pts2, M)
    #FRansac = eightpoint(pts1, pts2, M)
    pts1Valid = pts1
    pts2Valid = pts2
    pts1Valid = pts1[np.where(inliers == True)]
    pts2Valid = pts2[np.where(inliers == True)]
    #helper.displayEpipolarF(im1, im2, FRansac)
    #FEight = eightpoint(pts1, pts2, M)
    #helper.displayEpipolarF(im1, im2, FEight)
    '''
    pts1Valid = pts1[np.where(inliers == True)]
    pts2Valid = pts2[np.where(inliers == True)]
    numPoints = pts1.size // 2
    randPerm = np.random.permutation(numPoints)
    randPoints = randPerm[0:7]
    Farray = sevenpoint(pts1[randPoints,:], pts2[randPoints,:], M)
    helper.displayEpipolarF(im1, im2, Farray[0])
    outFile = 'q2_2.npz'
    np.savez(outFile, F = Farray[0], pts1 = pts1[randPoints,:], pts2 = pts2[randPoints,:], M = M)
    '''
    KIntrinsic = np.load('../data/intrinsics.npz')
    KIntrinsic1 = KIntrinsic['K1']
    KIntrinsic2 = KIntrinsic['K2']

    E = essentialMatrix(FRansac, KIntrinsic1, KIntrinsic2)

    ExtrinsicMList = helper.camera2(E)
    ProjMatrix1 = np.zeros([3,4])
    ProjMatrix1[0,0] = 1
    ProjMatrix1[1,1] = 1
    ProjMatrix1[2,2] = 1

    maxCount = -1
    minError = 999999999
    bestC2 = np.zeros([3,4])
    P_init = np.zeros((pts1Valid.shape[0], 3))
    for i in range(ExtrinsicMList.shape[2]) :
        M1 = ProjMatrix1
        M2 = ExtrinsicMList[:,:,i]
        [W1, err1] = triangulate(np.matmul(KIntrinsic1,M1), pts1Valid, np.matmul(KIntrinsic2,M2), pts2Valid)
        zIndicesCam1 = W1[:,2]
        validZValCam1 = np.where(zIndicesCam1 > 0)

        Rinv = np.linalg.inv(M2[:,0:3])
        tInv = -np.matmul(np.linalg.inv(M2[:,0:3]), M2[:,3])

        M2_new = ProjMatrix1
        M1_new = np.zeros((3,4))
        M1_new[:,0:3] = Rinv
        M1_new[:,3] = tInv

        [W2, err2] = triangulate(np.matmul(KIntrinsic1, M1_new), pts1Valid, np.matmul(KIntrinsic2, M2_new), pts2Valid)
        zIndicesCam2 = W2[:,2]
        validZValCam2 = np.where(zIndicesCam2 > 0)
        validZBothCam = np.intersect1d(validZValCam1, validZValCam2)
        validCount = validZBothCam.size
        if validCount > maxCount :
            maxCount = validCount
            maxError = err1
            bestM2 = M2
            P = W1
    M2_init = bestM2
    #print (maxCount)
    C2_init = np.matmul(KIntrinsic2, M2_init)
    P_init = P

    M2Final, wFinal = bundleAdjustment(KIntrinsic1, M1, pts1Valid, KIntrinsic2, M2_init, pts2Valid, P_init)

    #print (M2_init)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, xmax = np.min(P_init[:, 0]), np.max(P_init[:, 0])
    ymin, ymax = np.min(P_init[:, 1]), np.max(P_init[:, 1])
    zmin, zmax = np.min(P_init[:, 2]), np.max(P_init[:, 2])

    ax.set_xlim3d(xmin, xmax)
    ax.set_xlabel('x')
    ax.set_ylim3d(ymin, ymax)
    ax.set_ylabel('y')
    ax.set_zlim3d(zmin, zmax)
    ax.set_zlabel('z')

    ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], c='b', marker='o')
    plt.show()

    #print (M2Final, wFinal)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, xmax = np.min(wFinal[:, 0]), np.max(wFinal[:, 0])
    ymin, ymax = np.min(wFinal[:, 1]), np.max(wFinal[:, 1])
    zmin, zmax = np.min(wFinal[:, 2]), np.max(wFinal[:, 2])

    ax.set_xlim3d(xmin, xmax)
    ax.set_xlabel('x')
    ax.set_ylim3d(ymin, ymax)
    ax.set_ylabel('y')
    ax.set_zlim3d(zmin, zmax)
    ax.set_zlabel('z')

    ax.scatter(wFinal[:, 0], wFinal[:, 1], wFinal[:, 2], c='b', marker='o')
    plt.show()