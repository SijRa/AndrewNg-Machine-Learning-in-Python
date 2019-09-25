import mat4py.loadmat as load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

data = load('ex6data1.mat')
data2 = load('ex6data2.mat')
data3 = load('ex6data3.mat')

X1 = np.array(data['X'])
y1 = np.array(data['y']).flatten()

X2 = np.array(data2['X'])
y2 = np.array(data2['y']).flatten()

X3 = np.array(data3['X'])
y3 = np.array(data3['y']).flatten()
X3val = np.array(data3['Xval'])
y3val = np.array(data3['yval']).flatten()

# Gaussian Kernel
def GaussianKernel(sigma):
    """
    Returns similarity matrix
    """
    def GaussianFunction(X1, X2):
        """
        Calculate similarity between 2 points
        """
        matrix = np.zeros((X1.shape[0], X2.shape[0]))
        for i, xi in enumerate(X1):
            for j, xii in enumerate(X2):
                u = xi.flatten()
                v = xii.flatten()
                matrix[i, j] = np.exp(np.divide(-np.sum((u - v)**2), 2*sigma**2))
        return matrix
    return GaussianFunction

# Linear SVM
LinSVC = LinearSVC(C=1.0)
model1 = LinSVC.fit(X1, y1)
theta1 = model1.coef_
intercept1 = model1.intercept_

# SVM with custom kernel (Gaussian)
sigma = 0.05
GauSVC = SVC(kernel=GaussianKernel(sigma), C=1.0)
model2 = GauSVC.fit(X2, y2)

# Example dataset 1
df1 = pd.DataFrame([X1[:,0],X1[:,1], y1]).T
df1.columns = ['X₁', 'X₂', 'y']

circle1 = df1[df1['y'] == 0]
plus1 = df1[df1['y'] == 1]

plt.figure()
plt.title("Figure 1")
xAxis = np.array(np.linspace(np.min(X1), np.max(X1), 100))
yAxis = [-np.divide(theta1[:,0] * i + intercept1, theta1[:,1]) for i in xAxis]
plt.scatter(circle1['X₁'], circle1['X₂'], marker='.', c='red')
plt.scatter(plus1['X₁'], plus1['X₂'], marker='+', c='black')
plt.plot(xAxis, yAxis, c='blue')

# Example dataset 2
df2 = pd.DataFrame([X2[:,0],X2[:,1], y2]).T
df2.columns = ['X₁', 'X₂', 'y']

circle2 = df2[df2['y'] == 0]
plus2 = df2[df2['y'] == 1]

plt.figure()
plt.title("Figure 2")
xAxis = np.array(np.linspace(np.min(X2), np.max(X2), 100))
plt.scatter(circle2['X₁'], circle2['X₂'], marker='.', c='red')
plt.scatter(plus2['X₁'], plus2['X₂'], marker='+', c='black')

plotx1 = np.linspace(np.min(X2[:,0]), np.max(X2[:,0]), 100)
plotx2 = np.linspace(np.min(X2[:,1]), np.max(X2[:,1]), 100)

mX1, mX2 = np.meshgrid(plotx1, plotx2)
vals = np.zeros(mX1.shape)
for i in range(mX1.shape[1]):
    x = np.array([mX1[:,i], mX2[:,i]]).T
    vals[:,i] = model2.predict(x)
plt.contour(mX1, mX2, vals, levels=[0,0], colors='blue')

# Example dataset 3
df3 = pd.DataFrame([X3[:,0],X3[:,1], y3]).T
df3.columns = ['X₁', 'X₂', 'y']

circle3 = df3[df3['y'] == 0]
plus3 = df3[df3['y'] == 1]

cValues = np.array([0.5, 1, 2])

plot2x1 = np.linspace(np.min(X3[:,0]), np.max(X3[:,0]), 100)
plot2x2 = np.linspace(np.min(X3[:,1]), np.max(X3[:,1]), 100)

m2X1, m2X2 = np.meshgrid(plot2x1, plot2x2)

plt.figure()
plt.title("Figure 3")
plt.scatter(circle3['X₁'], circle3['X₂'], marker='.', c='red')
plt.scatter(plus3['X₁'], plus3['X₂'], marker='+', c='black')

for cValue in cValues:
    model3 = SVC(C=cValue, kernel=GaussianKernel(0.1)).fit(X3, y3)
    accuracy = model3.score(X3val, y3val)
    vals2 = np.zeros(m2X1.shape)
    plt.figure()
    plt.scatter(circle3['X₁'], circle3['X₂'], marker='.', c='red')
    plt.scatter(plus3['X₁'], plus3['X₂'], marker='+', c='black')
    plt.title("Figure 3" + "\n Mean Accuracy: " + str(accuracy) + "\nC = " + str(cValue) + " Sigma = " + str(0.1))
    for i in range(m2X1.shape[1]):
        x = np.array([m2X1[:,i], m2X2[:,i]]).T
        vals2[:,i] = model3.predict(x)
    plt.contour(m2X1, m2X2, vals2, levels=[0,0], colors='blue')
plt.show()