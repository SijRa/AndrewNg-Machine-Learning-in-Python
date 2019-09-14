import mat4py.loadmat as load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import optimize

DisplayData = True
DisplayLearningCurve = True
FindOptimalLambda = True

data = load('ex5data1.mat')

X = np.array(data['X'])
# Bias
X = np.insert(X, 0, 1, axis=1)
y = np.array(data['y'])

X_val = np.array(data['Xval'])
# Bias
X_val = np.insert(X_val, 0, 1, axis=1)
y_val = np.array(data['yval'])

X_test = np.array(data['Xtest'])
# Bias
X_test = np.insert(X_test, 0, 1, axis=1)
y_test = np.array(data['ytest'])

lmbda = 0.8
p = 8

def RegularizedCostFunction(theta, X, y, lmbda):
    m = X.shape[0]
    return np.divide(np.sum((np.dot(X, theta) - y) ** 2), 2 * m) + np.divide(lmbda * np.sum(theta[1:] ** 2), 2 * m)

def RegularizedGradient(theta, X, y, lmbda):
    m = X.shape[0]
    gradient = np.zeros((theta.shape))
    for j in range(theta.shape[0]):
        gradient[j] = np.divide(np.sum(np.dot((np.dot(X, theta) - y), X.T[j])), m)
        if(j >= 1):
            gradient[j] += np.divide(lmbda * theta[j], m)
    return gradient

def PolynomialFeatures(X, degrees):
    # Assumes bias already added    
    newFeatures = X.copy().reshape((X.shape[0],2))
    for i in range(2, degrees + 1):
        newFeatures = np.insert(newFeatures, i, newFeatures[:,1] ** i, axis=1)
    return newFeatures

def Train(X, y, lmbda):
    theta = np.zeros((X.shape[1], 1))
    return optimize.fmin_cg(f=RegularizedCostFunction, x0=theta, fprime=RegularizedGradient, args=(X, y.flatten(), lmbda)).reshape((X.shape[1],1))

def FeatureNormalisation(X):
    # Assumes bias already added
    normalised_X = X.copy()
    mu = []
    sigma = []
    for i in range(1, X.shape[1]):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        normalised_X[:,i] = np.divide(X[:,i] - mean, std)
        mu.append(mean)
        sigma.append(std)
    return normalised_X, np.asarray(mu), np.asarray(sigma)

X_polynomials = PolynomialFeatures(X, p)
X_normalised, mu, sigma = FeatureNormalisation(X_polynomials)

# Normalisation
X_val_poly = PolynomialFeatures(X_val, p)
X_val_poly[:,1:] -= mu
X_val_poly[:,1:] /= sigma

# Normalisation
X_test_poly = PolynomialFeatures(X_test, p)
X_test_poly[:,1:] -= mu
X_test_poly[:,1:] /= sigma

optimal_theta = Train(X_normalised, y, lmbda)

if(DisplayData):
    plt.figure()
    plt.scatter(X[:,1], y, marker='x', c='red')
    # x-axis
    x = np.arange(np.min(X) - 15, np.max(X) + 25, 0.05)
    x = x.reshape((x.shape[0],1))
    # Bias
    x = np.insert(x, 0, 1, axis=1)
    # Generate polynomial features
    x_poly = PolynomialFeatures(x, p)
    # Normalisation
    x_poly[:,1:] -= mu
    x_poly[:,1:] /= sigma
    plt.plot(x[:,1:], np.dot(x_poly, optimal_theta), c='blue', linestyle='dashed')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of the dam')
    
if(DisplayLearningCurve):
    m = X_normalised.shape[0]
    training_CostHistory = []
    validation_CostHistory = []
    for i in range(1, m + 1):
        theta = Train(X_normalised[:i], y[:i], lmbda)
        training_CostHistory.append(RegularizedCostFunction(theta, X_normalised[:i], y[:i], lmbda))
        validation_CostHistory.append(RegularizedCostFunction(theta, X_val_poly, y_val, lmbda))
    plt.figure()
    plt.plot(np.arange(1, m + 1), training_CostHistory, c='blue', label='Training')
    plt.plot(np.arange(1, m + 1), validation_CostHistory, c='green', label='Cross Validation')
    plt.legend()
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")

if(FindOptimalLambda):
    lambdaValues = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    training_CostHistory = []
    validation_CostHistory = []
    for value in lambdaValues:
        theta = Train(X_normalised, y, value)
        training_CostHistory.append(RegularizedCostFunction(theta, X_normalised, y, value))
        validation_CostHistory.append(RegularizedCostFunction(theta, X_val_poly, y_val, value))
    plt.figure()
    plt.plot(lambdaValues, training_CostHistory, c='blue', label='Training')
    plt.plot(lambdaValues, validation_CostHistory, c='green', label='Cross Validation')
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylabel("Error")

if DisplayData or DisplayLearningCurve or FindOptimalLambda:
    plt.show()