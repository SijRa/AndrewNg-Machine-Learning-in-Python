import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

## MULTIVARIAYE LINEAR REGRESSION ##

Optimise_Theta = False

house_data = pd.read_csv('ex1data2.txt', header=None)
house_data.columns = ['Size(SqFt)','Bedroom', 'Price($)']

original_house_data = house_data.copy()

# Feature Normalisation
sizeMean = house_data['Size(SqFt)'].mean()
sizeStd =  house_data['Size(SqFt)'].std()

bedroomMean = house_data['Bedroom'].mean()
bedroomStd =  house_data['Bedroom'].std()

priceMean = house_data['Price($)'].mean()
priceStd =  house_data['Price($)'].std()

house_data['Size(SqFt)'] = np.divide(house_data['Size(SqFt)'] - sizeMean, sizeStd)
house_data['Bedroom'] = np.divide(house_data['Bedroom'] - bedroomMean, bedroomStd)
house_data['Price($)'] = np.divide(house_data['Price($)'] - priceMean, priceStd)

X = pd.DataFrame(house_data[['Size(SqFt)','Bedroom']])
y = pd.DataFrame(house_data['Price($)'])

# Columns of 1s
Ones = [x/x for x in range(1,len(X.index) + 1)]
dfOnes = pd.DataFrame(Ones)
X.insert(0, column='Constant', value=dfOnes)

# Gradient Descent
theta = np.zeros((len(X.columns),1))
iterations = 450
alpha = 0.02

costFunction_History = []
def CostFunction(X, y, theta):
    J = 0
    m = len(y.index)
    totalError = 0
    for i in range(m):
        prediction = np.dot(theta.transpose(), X.iloc[i].values)
        errorSqrd = (prediction - y.iloc[i].values)**2
        totalError += errorSqrd
    J = np.divide(totalError,2*m)
    costFunction_History.append(J)
    return J

def SumFunct(X, y, theta, j, m):
    total = 0
    for i in range(m):
        prediction = np.dot(theta.transpose(),X.iloc[i].values)
        error = prediction - y.iloc[i]
        if j == 0:
            total += error
        elif j == 1:
            total += error * X.iloc[i].values[1]
        elif j == 2:
            total += error * X.iloc[i].values[2]
    return total

def GradientDecent(X, y, theta, alpha, iterations):
    m = len(y.index)
    tempTheta = theta
    for i in range(iterations):
        tempTheta[0] = theta[0] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 0, m)
        tempTheta[1] = theta[1] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 1, m)
        tempTheta[2] =  theta[2] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 2, m)
        theta = tempTheta
        print("Iteration: " + str(i) + "\nJ(0) = " + str(CostFunction(X, y, theta)))
        print("Theta values:\n", theta)
    return theta


## OPTIMISE θ ##
if (Optimise_Theta):
    maxCost = CostFunction(X, y, theta)
    theta = GradientDecent(X, y, theta, alpha, iterations)

    # Cost function against iterations
    plt.plot(np.arange(0,iterations+1),costFunction_History)
    plt.xlabel("Iterations")
    plt.xticks(np.arange(0, iterations + 50, 50))
    plt.ylabel("J(θ)")
    plt.show()


## FINDING θ WITH NORMAL EQUATION ##
Xarray = X.to_numpy()
yarray = y.to_numpy()
transposedX = Xarray.transpose()

normalEq_theta = np.matmul(np.linalg.inv(np.matmul(transposedX, Xarray)), np.matmul(transposedX, yarray))
print(CostFunction(X, y, normalEq_theta))