import mat4py.loadmat as load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import optimize

display_Digits = False
display_Hidden_Layer = True
CheckGradient = False

# Extract and format data from MATLAB files to Pandas dataframes
data = load('ex4data1.mat')
data_theta = load('ex4weights.mat')

# Digits
training_examples = []
for value in data['X']:
    training_examples.append(value)
df_digits = pd.DataFrame(training_examples)

# Labels
training_labels = []
for value in data['y']:
    training_labels.append(value)
df_labels = pd.DataFrame(training_labels)

# Pre-trained weights Weights
theta1 = []
theta2 = []
for value in data_theta['Theta1']:
    theta1.append(value)
for value in data_theta['Theta2']:
    theta2.append(value)
df_theta1 = pd.DataFrame(theta1)
df_theta2 = pd.DataFrame(theta2)
theta1 = df_theta1.to_numpy().flatten() # Vectorized
theta2 = df_theta2.to_numpy().flatten() # Vectorized

X = df_digits.copy().to_numpy()
# Bias added
X = np.insert(X, 0, 1, axis=1)

y = df_labels.copy()
lam = 1

input_layer_size = 400 # 20x20 images of digits
hidden_layer_size = 25
num_labels = 10

nn_params = np.concatenate((theta1,theta2))

# Vectorize labels
yVectorsEmpty = []
yTemp = y.replace({1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9})
for value in yTemp[0]:
    vector = np.zeros(num_labels)
    vector[value] = 1
    yVectorsEmpty.append(vector)
yVectors = np.array(yVectorsEmpty).reshape((y.shape[0],num_labels))

def SigmoidFunction(value):
    return np.divide(1, 1 + np.exp(-value))

def SigmoidGradient(z):
    value = SigmoidFunction(z)
    return np.multiply(value, 1 - value) 

def RandomInitalWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
    Theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, hidden_layer_size + 1)
    m = X.shape[0]
    # FeedForward Propagation
    hidden_layer = pd.DataFrame(SigmoidFunction(np.dot(X, Theta1.T))).to_numpy()
    # Bias added
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)
    output_layer = SigmoidFunction(np.dot(hidden_layer, Theta2.T))
    # Regularised Cost
    return np.divide(np.sum(-y * np.log(output_layer) - (1 - y) * np.log(1 - output_layer)), m) + np.divide(lam * (np.sum(Theta1[:,1:] ** 2) + np.sum(Theta2[:,1:] ** 2)), 2 * m)

def NumericalGradient(testParam):
    e = 1e-4
    gradApprox = np.zeros(testParam.shape)
    for i in range(testParam.shape[0]):
        thetaPlus = testParam.copy()
        thetaPlus[i] = thetaPlus[i] + e
        thetaMinus = testParam.copy()
        thetaMinus[i] = thetaMinus[i] - e
        gradApprox[i] = np.divide(
            nnCostFunction(thetaPlus, input_layer_size, hidden_layer_size, num_labels, X, yVectors, lam) - nnCostFunction(thetaMinus, input_layer_size, hidden_layer_size, num_labels, X, yVectors, lam),
            2 * e)
    return gradApprox

def nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
    Theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, hidden_layer_size + 1)
    m = X.shape[0]
    # FeedForward Propagation
    Z_2 = np.dot(Theta1, X.T)
    A_2 = SigmoidFunction(Z_2)
    A_2 = np.insert(A_2, 0, 1, axis=0) # Bias
    # Hidden Layer
    Z_3 = np.dot(Theta2, A_2)
    A_3 = SigmoidFunction(Z_3)
    # Backpropagation
    d_3 = A_3 - y.T
    d_2 = np.dot(Theta2.T, d_3) * np.insert(SigmoidGradient(Z_2), 0, 1, axis=0)
    D_2 = np.dot(d_3, A_2.T)
    D_1 = np.dot(d_2[1:], X)
    D_2 /= m
    D_1 /= m
    # Regularisation
    Grad2_0 = D_2[:,0]
    Grad2 = D_2 + np.divide(Theta2, m)
    Grad2[:,0] = Grad2_0
    Grad1_0 = D_1[:,0]
    Grad1 = D_1 + np.divide(Theta1, m)
    Grad1[:,0] = Grad1_0
    return np.concatenate((Grad1.flatten(), Grad2.flatten()))

def Train(parameters):
    optimialParameters = optimize.fmin_cg(f=nnCostFunction, x0=parameters, fprime=nnGradient, args=((input_layer_size, hidden_layer_size, num_labels, X, yVectors, lam)))
    return optimialParameters

def Predict(parameters, X):
    Theta1 = parameters[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = parameters[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, hidden_layer_size + 1)
    m = X.shape[0]
    hidden_layer = SigmoidFunction(np.dot(X, Theta1.T))
    # Bias Added
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)
    output_layer = SigmoidFunction(np.dot(hidden_layer, Theta2.T))
    df = pd.DataFrame(data=output_layer)
    return (df.idxmax(axis=1) + 1).to_numpy() # Predictions changed from 0-9 to 1-10

inital_Theta1 = RandomInitalWeights(input_layer_size, hidden_layer_size)
inital_Theta2 = RandomInitalWeights(hidden_layer_size, num_labels)
inital_nn_params = np.concatenate((inital_Theta1.flatten(), inital_Theta2.flatten()))

optimial_parameters = Train(inital_nn_params)

preds = Predict(optimial_parameters, X)
correct = (y.to_numpy().flatten() == preds).astype(int)
print("Accuracy: " + str(np.sum(correct)*100/X.shape[0]) + "%")

if(CheckGradient):
    checkGrad = NumericalGradient(inital_nn_params)
    checkTheta1 = checkGrad[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size + 1)
    checkTheta2 = checkGrad[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, hidden_layer_size + 1)

if(display_Hidden_Layer):
    Hidden_Layer_Theta = optimial_parameters[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size + 1)[:,1:]
    fig = plt.figure(figsize=(10, 10))
    columns = 5
    rows = 5
    for i in range(hidden_layer_size):
        pixels = Hidden_Layer_Theta[i].reshape((20,20))
        fig.add_subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.imshow(pixels, cmap='gray')
    plt.show()

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