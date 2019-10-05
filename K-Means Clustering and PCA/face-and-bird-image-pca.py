import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data = sio.loadmat('ex7faces.mat')
data_bird = sio.loadmat('bird_small.mat')

X = data['X']
A = np.divide(data_bird['A'], 255)

image_num = 36

def DisplayFaces(X, image_num, title=None):
    fig = plt.figure()
    plt.axis('off')
    if title:
        plt.title(title)
    for i in range(1,image_num+1):
        fig.add_subplot(np.sqrt(image_num), np.sqrt(image_num), i)
        plt.axis('off')
        image_pixels = X[i,:].reshape(32,32)
        rotated = ndimage.rotate(image_pixels, -90)
        plt.imshow(rotated, cmap='gray')

def FeatureNormalisation(X):
    normalised_X = X.copy()
    mu = []
    sigma = []
    for i in range(X.shape[1]):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        normalised_X[:,i] = np.divide(X[:,i] - mean, std)
        mu.append(mean)
        sigma.append(std)
    return normalised_X, np.asarray(mu), np.asarray(sigma)

def PCA(X):
    m, n = X.shape
    sigma = np.divide(np.dot(X.T,X),m)
    U, S, V = np.linalg.svd(sigma)
    return U, S # Eigen vectors and Eigen values

def ProjectData(X, U, K):
    return np.dot(U[:,:K].T, X.T)

def RecoverData(Z, U, K):
    return np.matmul(U[:,:K], Z).T

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

# Face PCA
X_norm, mu, sigma = FeatureNormalisation(X)
U, S = PCA(X_norm)

K = 50
Z = ProjectData(X_norm, U, K)
X_rec = RecoverData(Z, U, K)

# Bird image PCA
K_bird = 16
max_iters = 10
img_size = A.shape
X_bird = np.reshape(A, (A.shape[0] * A.shape[1], 3))
centroids, idx = K_Means(X_bird, K_bird, max_iters)

X_bird_norm, mu, sigma = FeatureNormalisation(X_bird)
U_bird, S = PCA(X_bird_norm)
Z = ProjectData(X_bird_norm, U_bird, 2)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
colors = ['black', 'silver', 'lightcoral', 'firebrick', 'coral', 'bisque', 'gold', 'olive', 'palegreen', 
'mediumseagreen', 'turquoise', 'teal', 'deepskyblue', 'slategray', 'royalblue', 'darkorchid', 'magenta', 'lightpink']
for i in range(K_bird):
    index = np.argwhere(idx.flatten()==i).flatten()
    ax.scatter(X_bird[index,0], X_bird[index,1], X_bird[index,2], cmap=colors[i], marker='.')

plt.figure()
plt.title("Pixel dataset plotted in 2D - reduced dimensions using PCA")
for i in range(K_bird):
    index = np.argwhere(idx.flatten()==i).flatten()
    plt.scatter(X_bird_norm[index,0], X_bird_norm[index,1], cmap=colors[i], marker='.')

DisplayFaces(X, image_num, 'Original')
DisplayFaces(U[:,:image_num+1].T, image_num, 'Images from eigenvectors')
DisplayFaces(X_norm, image_num, 'Normalised Images')
DisplayFaces(X_rec, image_num, 'PCA ' + str(K))

plt.show()