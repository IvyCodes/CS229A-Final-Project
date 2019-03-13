import scipy.io.wavfile as wav
from python_speech_features import mfcc
import pandas as pd
import numpy as np
from scipy import linalg

'''

'''

def sigmoid(z):
	return 1/(1+np.exp(-z))

def cost_function(theta, X, y, lamb):
	m = y.size
	h = sigmoid(X.dot(theta))
	r = -y.T.dot(np.log(h))
	l = (1-y).T.dot(np.log(1-h))

	J = (r-l)/m
	grad = X.T.dot(h-y)/m

	theta[0] = 0
	cost_reg = (theta.T.dot(theta)) * lamb/(2*m)
	J+=cost_reg

	grad_reg = theta*lamb/m
	grad += grad_reg

	# print(h)
	# print(J)
	# print(grad)
	return (J,grad)

def update(theta, grad, alpha):
	return theta - alpha.dot(grad)


file = r'labels.csv'
df = pd.read_csv(file)

'''
i = 0
for index, row in df.iterrows():
    itemid = row['itemid']
    hasbird = row['hasbird']
    print(str(itemid) + " " + str(hasbird))
    i += 1
    if i >= 20:
    	break
 '''

# X = np.array([[1,1],[3,4]])
# theta = np.array([[1],[2],[-2]])
# X = np.insert(X, 0, 1, axis = 1)
# y = np.array([[1],[0]])
# lamb = .5
# cost_function(theta, X, y, lamb)

m = 10

y = np.array([df['hasbird'].tolist()[0:m]]).T
X = []

audio_list = df['itemid'].tolist()[0:m]
for id in audio_list:
	(rate,sig) = wav.read("./wav/" + str(id) + ".wav")
	print(rate)
	mfcc_feat = mfcc(sig,rate)
	print(np.size(mfcc_feat, axis = 0))
	X.insert(mfcc_feat)

print(X)

theta = np.array([[0]*X.size])
print(y)
print(theta)
#print(y[0:20])