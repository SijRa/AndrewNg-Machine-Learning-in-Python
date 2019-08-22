import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ShowGraph = True

df = pd.read_csv("ex1data1.txt", header=None)
df.columns = ['PopulationOfCity', 'ProfitOfFoodTruck']


# Add column of 1s to X 
y = pd.DataFrame(df['ProfitOfFoodTruck'])
X = pd.DataFrame(df['PopulationOfCity'])
Ones = [x/x for x in range(1,len(X.index) + 1)]
dfOnes = pd.DataFrame(Ones)
X.insert(0, column='Constant', value=dfOnes)

theta = np.zeros((2,1))

iterations = 1500
alpha = 0.02

costFunction_vals = []
def CostFunction(X, y, theta):
    J = 0
    m = len(y.index)
    totalError = 0
    for i in range(m):
        prediction = np.dot(theta.transpose(), X.iloc[i].values)
        errorSqrd = (prediction - y.iloc[i].values)**2
        totalError += errorSqrd
    J = np.divide(totalError,2*m)
    return J

# Inital J value
costFunction_vals.append(CostFunction(X, y, theta))

def SumFunct(X, y, theta, j, m):
    total = 0
    for i in range(m):
        prediction = np.dot(theta.transpose(),X.iloc[i].values)
        error = prediction - y.iloc[i]
        if j == 0:
            total += error
        elif j == 1:
            total += error * X.iloc[i].values[1]
    return total

def GradientDecent(X, y, theta, alpha, iterations):
    m = len(y.index)
    tempTheta = theta
    for i in range(iterations):
        tempTheta[0] = theta[0] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 0, m)
        tempTheta[1] = theta[1] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 1, m) 
        theta = tempTheta
        print("J(0) =", CostFunction(X, y, theta))
        print("Theta values:\n", theta)
        return theta

#theta = GradientDecent(X, y, theta, alpha, iterations)

# Best theta values
theta[0] = -3.87964303
theta[1] = 1.1914183

yAxis = [np.dot(theta.transpose(),X.iloc[i].values) for i in range(len(X.index))]
xAxis = [X.iloc[i].values[1] for i in range(len(X.index))]

if(ShowGraph):
    plt.scatter(df['PopulationOfCity'],df['ProfitOfFoodTruck'], c='red', marker='x', label='Training data')
    plt.plot(xAxis,yAxis, c='blue', label='Linear regression')
    plt.legend(loc='center', bbox_to_anchor=(0.8,0.2))
    plt.xticks(np.arange(4,25,2))
    plt.ylim(-5,25)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()