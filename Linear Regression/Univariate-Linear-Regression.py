import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

## UNIVARIATE LINEAR REGRESSION ##

Optimise_Theta = True

LinearRegressionGraph = True
GradientDescentGraph = True

df = pd.read_csv("ex1data1.txt", header=None)
df.columns = ['PopulationOfCity', 'ProfitOfFoodTruck']

# Add column of 1s to X 
y = pd.DataFrame(df['ProfitOfFoodTruck'])
X = pd.DataFrame(df['PopulationOfCity'])
Ones = [x/x for x in range(1,len(X.index) + 1)]
dfOnes = pd.DataFrame(Ones)
X.insert(0, column='Constant', value=dfOnes)

# Gradient Descent
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.02

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

def GradientDescent(X, y, theta, alpha, iterations):
    m = len(y.index)
    tempTheta = theta
    for i in range(iterations):
        tempTheta[0] = theta[0] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 0, m)
        tempTheta[1] = theta[1] - alpha * np.divide(1,m) * SumFunct(X, y, theta, 1, m) 
        theta = tempTheta
        print("Iteration: " + str(i) + "\nJ(0) = " + str(CostFunction(X, y, theta)))
        print("Theta values:\n", theta)
    return theta

# Linear Regression
if(LinearRegressionGraph):
    plt.scatter(df['PopulationOfCity'],df['ProfitOfFoodTruck'], c='red', marker='x', label='Training data')
    xAxis = [X.iloc[i].values[1] for i in range(len(X.index))]
    yAxis = [np.dot(theta.transpose(),X.iloc[i].values) for i in range(len(X.index))]
    plt.plot(xAxis,yAxis, c='blue', label='Linear regression')
    plt.legend(loc='center', bbox_to_anchor=(0.8,0.2))
    plt.xticks(np.arange(4,25,2))
    plt.ylim(-5,25)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")

# Cost Function Mapping
if(GradientDescentGraph):
    theta_0 = np.linspace(-10, 10, 100)
    theta_1 = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta_0),len(theta_1)))
    count = 0
    for i in range(0,len(theta_0)):
        for j in range(0, len(theta_1)):
            t = np.array([theta_0[i], theta_1[j]])
            J_vals[i,j] = CostFunction(X, y, t)
            count+=1
            print("Percentage complete: " + str(round((np.divide(count,10000) * 100), 2)) +"%")
    X_plot, Y_plot = np.meshgrid(theta_0, theta_1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X_plot, Y_plot, J_vals, color='green')
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_xlabel('θ₀')
    ax.set_ylabel('θ₁')
    ax.set_zlabel('J(θ₀θ₁)')
    ax.plot_surface(X_plot, Y_plot, J_vals, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.plot(theta[1], theta[0], CostFunction(X, y, theta), label='Optimal θ', markeredgecolor='red', marker='x', markersize=5)
    ax.legend()

## OPTIMISE θ ## 
if (Optimise_Theta):
    theta = GradientDescent(X, y, theta, alpha, iterations)
    if(LinearRegressionGraph or GradientDescentGraph):
        plt.show()