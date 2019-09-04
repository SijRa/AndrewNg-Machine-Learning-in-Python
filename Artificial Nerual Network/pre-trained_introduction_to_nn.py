import mat4py.loadmat as load
import pandas as pd
import numpy as np

# Extract and format data from MATLAB files to Pandas dataframes
data = load('ex3data1.mat')
data_theta = load('ex3weights.mat')

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

# Bias Added
Ones = [np.divide(x,x) for x in range(1,df_digits.shape[0] + 1)]
dfOnes = pd.DataFrame(Ones)
df_digits.insert(0, column='Constant', value=dfOnes)

X = df_digits
y = df_labels

def SigmoidFunction(value):
    return np.divide(1, 1 + np.exp(-value))

def Predict(theta1, theta2, X):
    m = X.shape[0]
    hidden1 = pd.DataFrame(SigmoidFunction(np.dot(X, theta1.T)))
    Ones = [x/x for x in range(1, m + 1)]
    # Bias Added
    dfOnes = pd.DataFrame(Ones)
    hidden1.insert(0, column='Constant', value=dfOnes)
    hidden2 = SigmoidFunction(np.dot(hidden1, theta2.T))
    df = pd.DataFrame(data=hidden2)
    return df.idxmax(axis=1) + 1 # Predictions changed from 0-9 to 1-10

preds = Predict(df_theta1, df_theta2, X)
correct = (y.to_numpy().flatten() == preds.to_numpy()).astype(int)
print("Accuracy: " + str(np.sum(correct)*100/X.shape[0]) + "%")