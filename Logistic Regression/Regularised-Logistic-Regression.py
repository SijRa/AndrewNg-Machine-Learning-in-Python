import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df = pd.read_csv('ex2data2.txt', header=None)
df.columns = ['MicrochipTest1', 'MicrochipTest2', 'Pass']

ShowScatter = True

X1 = df['MicrochipTest1']
X2 = df['MicrochipTest2']

def MapFeatures(X1, X2, plotGraph):
    degree = 6
    if(plotGraph):
        out = np.ones(1)
    else:
        out = np.ones(X1.shape[0])[:,np.newaxis]
    for i in range(1,degree+1):
        for j in range(0,i+1):
            if(plotGraph):
                out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
            else:
                out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out

def SigmoidFunction(value):
    return np.divide(1, 1 + np.exp(-value))

def WeightedSum(Xi, theta):
    return np.matmul(theta.T, Xi.T)

def Probability(Xi, theta):
    return SigmoidFunction(WeightedSum(Xi, theta))

X = MapFeatures(X1, X2, plotGraph=False)
y = df['Pass']

lamb = 1
theta = np.zeros((X.shape[1],1))

def RegularisedCostFunction(theta, X, y):
    m = X.shape[0]
    return np.divide(np.sum(
        -y * np.log(Probability(X, theta).flatten()) - (1 - y) * np.log(1 - Probability(X, theta).flatten())
    ), m) + np.divide(lamb * np.sum(np.power(theta[1:], 2)), 2*m)

def Gradient(theta, X, y):
    m = X.shape[0]
    gradientTheta = np.zeros((theta.shape[0],1))
    constant = np.divide(1, m)  
    hx = Probability(X, theta).flatten()
    error = hx - y
    gradientTheta[0] = constant * np.dot(error, X.T[0])
    for j in range(1, theta.shape[0]):
        gradientTheta[j] = constant * np.dot(error, X.T[j]) + np.divide(lamb, m) * theta[j] 
    return gradientTheta

def Fit(X, y, theta):
    opt_weights = optimize.fmin_tnc(func=RegularisedCostFunction, x0=theta,
                  fprime=Gradient, args=(X, y))
    return opt_weights[0]

parameters = Fit(X, y, theta)
theta = parameters

if(ShowScatter):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(MapFeatures(u[i], v[j], plotGraph=True), theta)
    
    PassedTest = df[df['Pass'] == 1]
    FailTest = df[df['Pass'] == 0]
    plt.figure()
    plt.scatter(PassedTest['MicrochipTest1'], PassedTest['MicrochipTest2'], color='green', marker='.', label='Passed')
    plt.scatter(FailTest['MicrochipTest1'], FailTest['MicrochipTest2'], color='red', marker='.', label='Fail')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.contour(u,v,z,0, colors='blue', alpha=0.5)
    plt.legend()
    plt.show()