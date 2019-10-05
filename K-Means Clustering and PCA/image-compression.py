import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('bird_small.mat')['A']

A = np.divide(data, 255) # 255 bit pixel values squashed to range from 0-1
img_size = A.shape
X = np.reshape(A, (A.shape[0] * A.shape[1], 3))

K = 16
iterations = 10

def Init_Centroids(X, K):
    indexes = np.random.randint(0, X.shape[0], K)
    #np.random.shuffle(X)
    return X[indexes]

def FindClosestCentroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    for i, x in enumerate(X):
        min_dist = np.inf
        centroid_index = -1
        for j in range(K):
            distanceSq = np.sum((x - centroids[j]) ** 2)
            if(distanceSq < min_dist):
                min_dist = distanceSq
                centroid_index = j
        idx[i] = centroid_index
    return idx

def ComputeMeans(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((0, n))
    for k in range(K):
        x = X[np.argwhere(idx == k)[:,0]]
        mean = np.divide(np.sum(x, axis=0), x.shape[0]).reshape((1,n))
        centroids = np.append(centroids, mean, axis=0)
    return centroids

def K_Means(X, K, iterations, centroids=None):
    centroids = Init_Centroids(X, K)
    for i in range(iterations):
        idx = FindClosestCentroids(X, centroids)
        centroids = ComputeMeans(X, idx, K)
    return centroids, idx

final_centroids, idx = K_Means(X, K, iterations)

idx = FindClosestCentroids(X, final_centroids)

X_recovered = np.array([final_centroids[int(i)] for i in idx])
X_recovered = np.reshape(X_recovered, img_size)

plt.figure()
plt.title("Original")
plt.imshow(data)
plt.figure()
plt.title("Compressed")
plt.imshow(X_recovered)
plt.show()