import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

ScatterPlot = True

CostPlot = False
fmin_Optimise = True

gradientDecent = False
alpha = 0.001
iterations = 10000

main_data = pd.read_csv('ex2data1.txt', header=None)
main_data.columns = ['ExamScore1', 'ExamScore2', 'AdmissionStatus']

X = main_data[['ExamScore1','ExamScore2']]

# Add constant
Ones = [x/x for x in range(1,len(X.index) + 1)]
dfOnes = pd.DataFrame(Ones)
X.insert(0, column='Constant', value=dfOnes)

y = main_data['AdmissionStatus']

theta = np.zeros((len(X.columns),1))

X = X.to_numpy()
y = y.to_numpy()

def SigmoidFunction(value):
    return np.divide(1, 1 + np.exp(-value))

def WeightedSum(Xi, theta):
    return np.matmul(theta.T, Xi.T)

def Probability(Xi, theta):
    return SigmoidFunction(WeightedSum(Xi, theta))

cost_history = []
def CostFunction(theta, X, y):
    m = len(X)
    totalCost = 0
    for i in range(m):
        totalCost += -y[i] * np.log(Probability(X[i], theta)) - (1 - y[i]) * np.log(1 - Probability(X[i], theta))
    J = np.divide(totalCost, m)
    cost_history.append(J)
    return J

def GradientDescent(X, y, theta, alpha, iterations):
    m = len(X)
    CostFunction(theta, X, y) # Calculate inital cost
    for i in range(1,iterations+1):
        hx = Probability(X, theta)
        error = hx - y
        theta0 = theta[0] - np.divide(alpha, m) * np.dot(error, X.T[0])
        theta1 = theta[1] - np.divide(alpha, m) * np.dot(error, X.T[1])
        theta2 = theta[2] - np.divide(alpha, m) * np.dot(error, X.T[2])
        theta[0] = theta0
        theta[1] = theta1
        theta[2] = theta2
        print("Iteration: " + str(i) + "\nJ(0) = " + str(CostFunction(theta, X, y)))
        print("Theta values:\n", theta)
    return theta

def Gradient(theta, X, y):
    m = len(y)
    gradientTheta = np.zeros((len(theta),1))
    constant = np.divide(1, m)  
    hx = Probability(X, theta)
    error = hx - y
    gradientTheta[0] = constant * np.dot(error, X.T[0])
    gradientTheta[1] = constant * np.dot(error, X.T[1])
    gradientTheta[2] = constant * np.dot(error, X.T[2])
    return gradientTheta

def Fit(X, y, theta):
    opt_weights = optimize.fmin_tnc(func=CostFunction, x0=theta,
                  fprime=Gradient, args=(X, y))
    return opt_weights[0]

if fmin_Optimise:
    parameters = Fit(X, y, theta)
    theta = parameters
elif gradientDecent:
    theta = GradientDescent(X, y, theta, alpha, iterations)

## Plot data ##
if(ScatterPlot):
    xAxis = np.arange(main_data['ExamScore1'].min(),main_data['ExamScore1'].max())
    yAxis = []
    for i in xAxis:
        value = np.divide(-1 * (theta[0] + theta[1]*i), theta[2])
        yAxis.append(value)
    plt.figure()
    Addmission_0 = main_data[main_data['AdmissionStatus'] == 0]
    Addmission_1 = main_data[main_data['AdmissionStatus'] == 1] 
    plt.scatter(Addmission_0['ExamScore1'], Addmission_0['ExamScore2'], c='red', marker='.', label='Not admitted')
    plt.scatter(Addmission_1['ExamScore1'], Addmission_1['ExamScore2'], c='green', marker='.', label='Admitted')
    plt.xlabel('Exam Score 1')
    plt.ylabel('Exam Score 2')
    plt.plot(xAxis, yAxis, label='Descision boundary')
    plt.legend(loc='lower left')

if(CostPlot):
    plt.figure()
    xRange = np.arange(0, iterations + 1)
    plt.xlabel("Iterations")
    plt.ylabel("Jθ")
    plt.plot(xRange, cost_history)

if(ScatterPlot or CostPlot):
    plt.show()