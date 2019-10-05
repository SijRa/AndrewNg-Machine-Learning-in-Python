import numpy as np
import mat4py.loadmat as load
import matplotlib.pyplot as plt

data = load('ex7data2.mat')

X = np.array(data['X'])

K = 3 # 3 Centroids
iterations = 10

centroids = np.array([
    [3, 3],
    [6, 2],
    [8, 5]
])

def Init_Centroids(X, K):
    indexes = np.random.randint(0, X.shape[0], K)
    np.random.shuffle(X)
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
    centroid_movement = np.zeros((0, 2))
    centroid_movement = np.append(centroid_movement, centroids, axis=0)
    for i in range(iterations):
        idx = FindClosestCentroids(X, centroids)
        centroids = ComputeMeans(X, idx, K)
        centroid_movement = np.append(centroid_movement, centroids, axis=0)
    return centroids, idx, centroid_movement

final_centroids, idx, centroid_movement = K_Means(X, K, iterations)

red_x = X[np.argwhere(idx == 0)[:,0]]
blue_x = X[np.argwhere(idx == 1)[:,0]]
green_x = X[np.argwhere(idx == 2)[:,0]]

red_centroid_movement = np.array([centroid_movement[i] for i in range(centroid_movement.shape[0]) if i % K == 0])
blue_centroid_movement = np.array([centroid_movement[i] for i in range(centroid_movement.shape[0]) if i % K == 1])
green_centroid_movement = np.array([centroid_movement[i] for i in range(centroid_movement.shape[0]) if i % K == 2])

plt.figure(figsize=(10,8))
plt.title('K = ' + str(K))
plt.scatter(red_x[:,0], red_x[:,1], edgecolors='red', facecolors='none')
plt.plot(red_centroid_movement[:,0], red_centroid_movement[:,1], c='darkviolet')
plt.scatter(final_centroids[0,0], final_centroids[0,1], marker='X', c='black')

plt.scatter(blue_x[:,0], blue_x[:,1], edgecolors='blue', facecolors='none')
plt.plot(blue_centroid_movement[:,0], blue_centroid_movement[:,1], c='darkred')
plt.scatter(final_centroids[1,0], final_centroids[1,1], marker='X', c='black')

plt.scatter(green_x[:,0], green_x[:,1], edgecolors='green', facecolors='none')
plt.plot(green_centroid_movement[:,0], green_centroid_movement[:,1], c='darkorange')
plt.plot(final_centroids[2,0], final_centroids[2,1], marker='X', c='black',)
plt.show()