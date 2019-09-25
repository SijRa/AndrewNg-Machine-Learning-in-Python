import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import mat4py.loadmat as load

emailData = open('emailSample1.txt', 'r').read()

vocabList = pd.read_csv('vocab.txt',header=None)
vocabList.columns = ['Vocabs']

spamTrain = load('spamTrain.mat')
X_train = np.array(spamTrain['X']) # 4000 used to train
y_train = np.array(spamTrain['y']).flatten()

spamTest = load('spamTest.mat')
X_test = np.array(spamTest['Xtest']) # 1000 emails used to test
y_test = np.array(spamTest['ytest']).flatten()

## Clean and ready email for training ##

def PreprocessEmail(emailText):
    # Lower-casing
    text = emailText.lower()
    # Normalise numbers
    text = re.sub('[0-9]+', 'number', text)
    # Strip HTML
    text = re.sub('<[^<>]+>', ' ', text)
    # Normalise URL 
    text = re.sub('(http|https)://[^\s]*', 'httpaddr', text)
    # Normalise email
    text = re.sub('[^\s]+@[^\s]+', 'emailaddr', text)
    # Dollar sign
    text = re.sub('[$]+', 'dollar', text)
    return text

# Clean email
processedData = PreprocessEmail(emailData)

# Tokenize email
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(processedData)

# Stem words
ps = PorterStemmer()
stemmedTokens = [ps.stem(word) for word in tokens]

# Convert vocab to dictionary
vocab_array = np.array([re.findall(r'\w+', string) for string in vocabList['Vocabs']])
vocab_dict = {word : value for value,word in vocab_array}

mappedTokens = np.array([vocab_dict[token] for token in stemmedTokens if token in vocab_dict.keys()])

featureVector = np.zeros((vocab_array.shape[0], 1))

for value in mappedTokens:
    featureVector[int(value)] = 1

## Training and Testing Model ##

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

sigma = 0.1
GauSVC = SVC(kernel=GaussianKernel(sigma), C=1.0)
model_Gaus = GauSVC.fit(X_train, y_train)
score_Gaus_Train = model_Gaus.score(X_train, y_train)
score_Gaus_Test = model_Gaus.score(X_test, y_test)
print("--Gaussian SVC--")
print("\nTrain score:", score_Gaus_Train)
print("Test score:", score_Gaus_Test)

LinSVC = LinearSVC(C=1.0)
model_Lin = LinSVC.fit(X_train, y_train)
score_Lin_Train = model_Lin.score(X_train, y_train)
score_Lin_Test = model_Lin.score(X_test, y_test)
print("\n--Linear Kernel--")
print("\nTrain score:", score_Lin_Train)
print("Test score:", score_Lin_Test)

# Weights most predictive of spam
theta = model_Lin.coef_.flatten()
df_theta = pd.DataFrame(data=theta, columns=['Weight'])
df_theta = df_theta.sort_values(by=['Weight'], ascending=False)
indexValues = df_theta.index.values
numWords = 15
top_spam_indicators = [vocab_array[indexValues[i]][1] for i in range(numWords)]
print("Top 15 Spam Indicators: " + str(top_spam_indicators))