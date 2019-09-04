import mat4py.loadmat as load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import optimize

display_Digits = True

# Extract and format data from MATLAB files to Pandas dataframes
data = load('ex3data1.mat')

# Digits
training_examples = []
for value in data['X']:
    training_examples.append(value)
df_digits = pd.DataFrame(training_examples)

# Labels for digit examples
training_labels = []
for value in data['y']:
    training_labels.append(value)
df_labels = pd.DataFrame(training_labels)
df_labels.columns = ['Label']

num_labels = 10
X = df_digits
y = df_labels
lam = 0.1

def SigmoidFunction(value):
    return np.divide(1, 1 + np.exp(-value))

def WeightedSum(Xi, theta):
    return np.dot(Xi, theta)

def Probability(Xi, theta):
    return SigmoidFunction(WeightedSum(Xi, theta))

def lrCostFunction(theta, X, y, lam):
    m = X.shape[0]
    return np.divide(np.sum(
        -y * np.log(Probability(X, theta)) - (1 - y) * np.log(1 - Probability(X, theta))
    ), m) + np.divide(lam * np.sum(theta[1:] ** 2), 2 * m)

def lrGradient(theta, X, y, lam):
    m = X.shape[0]
    constant = np.divide(1, m)
    hx = Probability(X, theta)
    error = hx - y
    gradient = constant * np.matmul(X.T, error)
    reg_gradient = gradient + (np.divide(lam, m) * theta)
    # Bias θ₀ not regularised
    bias = gradient[0]
    reg_gradient[0] = bias
    return reg_gradient.flatten()

def OneVsAll(X, y, num_labels, lam):
    m = X.shape[0]
    n = X.shape[1]
    all_thetas = []
    # Columns of 1s
    Ones = [x/x for x in range(1,X.shape[0] + 1)]
    dfOnes = pd.DataFrame(Ones)
    X.insert(0, column='Constant', value=dfOnes)
    X = X.to_numpy()
    y = y.to_numpy()
    for i in range(1, num_labels + 1):
        theta = np.zeros(n + 1)
        y_labelled = (y == i).astype(int)
        opt_weights = optimize.fmin_cg(f=lrCostFunction, x0=theta, fprime=lrGradient, args=(X, y_labelled.flatten(), lam))
        all_thetas.append(opt_weights)
    return np.array(all_thetas).reshape((num_labels, n + 1))

def PredictOneVsAll(all_thetas, X):
    m = X.shape[0]
    probs = SigmoidFunction(np.matmul(X.to_numpy(), all_thetas.T))
    df = pd.DataFrame(data=probs)
    return df.idxmax(axis=1) + 1 # Predictions changed from 0-9 to 1-10

all_thetas = OneVsAll(X, y, num_labels, lam)
preds = PredictOneVsAll(all_thetas, X)
correct = (y.to_numpy().flatten() == preds.to_numpy()).astype(int)
print("Accuracy: " + str(np.sum(correct)*100/X.shape[0]) + "%")

if(display_Digits):
    fig = plt.figure(figsize=(10, 10))
    columns = 10
    rows = 10
    step = int(np.divide(df_digits.shape[0], columns*rows))
    index = 0
    if 'Constant' in df_digits:
        del df_digits['Constant']
    for i in range(1, columns*rows + 1):
        pixels = df_digits.iloc[index].to_numpy().reshape((20,20))
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(pixels, cmap='gray')
        index += step
    plt.show()