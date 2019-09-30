import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data = sio.loadmat('ex7data1.mat')

X = data['X']

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
    sigma = np.divide(np.matmul(X.T,X),m)
    U, S, V = np.linalg.svd(sigma)
    return U, S # Eigen vectors and Eigen values

def ProjectData(X, U, K):
    return np.dot(U[:,:K].T, X.T)

def RecoverData(Z, U, K):
    return (U[:,:K] * Z).T

X_norm, mu, sigma = FeatureNormalisation(X)
U, S = PCA(X_norm)

plt.figure()
plt.title("Original")
plt.scatter(X[:,0], X[:,1], edgecolors='blue', facecolors='none')
plt.quiver(*mu, U[:,0], U[:,1], color=['g', 'r'], scale=4)

plt.figure()
plt.title("Normalised")
origin = np.divide(np.sum(X_norm, axis=0),X_norm.shape[0])
plt.quiver(*origin, U[:,0], U[:,1], color=['g', 'r'], scale=4)
plt.scatter(X_norm[:,0], X_norm[:,1], edgecolors='red', facecolors='none')

K = 1
Z = ProjectData(X_norm, U, K)
X_recovered = RecoverData(Z, U, K)

plt.figure()
plt.title("Recovered data")
centre = np.divide(np.sum(X_recovered, axis=0),X_recovered.shape[0])
plt.quiver(*centre, U[:,0], U[:,1], color=['g', 'r'], scale=4)
plt.scatter(X_recovered[:,0], X_recovered[:,1], edgecolors='green', facecolors='none')

plt.show()