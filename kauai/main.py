import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

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
	return theta - grad.dot(alpha)

def train(X, y, lamb, alpha, num_iter):
	hist = []
	num_features=np.size(X, axis=1)
	theta = np.array([[0]*num_features]).T
	for i in range(num_iter):
		(J,grad) = cost_function(theta, X, y, lamb)
		hist.extend(J.tolist()[0])
		theta = update(theta, grad, alpha)

	return (theta,hist)

def create_features(winlen, winstep, file_name):
	(rate,sig) = wav.read(file_name)
	mfcc_feat = mfcc(sig,rate, numcep = 26, winlen = window_length, winstep = window_step)
	m = np.size(mfcc_feat, axis=0)
	X = np.insert(mfcc_feat, 0, 1, axis = 1)
	print('m = ' + str(m))
	return (X,m)

def create_labels(winlen, winstep, m, file_name):
	file = open(file_name, 'r')
	intervals = []
	y = [0]*m
	for line in file:
		(begin,end,_) = line.split('\t')
		intervals.append([float(begin),float(end)])

	cur_interval_pos = 0
	for i in intervals:
		begin_i = int(np.floor(i[0]/winstep))
		end_i = int(np.ceil((i[1]-winlen)/winstep))
		for pos in range(begin_i, end_i):
			y[pos] = 1
	return np.array([y]).T

def create_intervals(winlen, winstep, X, theta, out_file_name, threshold = 0.5, smooth_step = 20, ignore_step = 30):
	h = sigmoid(X.dot(theta))
	print('number of windows: ' + str(h.size))

	result = np.argwhere(h > threshold)[:,0]
	# print(result)
	intervals = []
	begin_i = result[0]
	end_i = result[0]
	for i in result:
		if i <= end_i + smooth_step:
			end_i = i
		else:
			if end_i-begin_i > ignore_step:
				begin_time = begin_i*winstep
				end_time = end_i * winstep
				intervals.append([str(begin_time), str(end_time), '\n'])
			begin_i = i
			end_i = i

	file = open(out_file_name, 'w+')
	for i in intervals:
		out_string = "\t".join(i)
		#print(i)
		file.write(out_string)
	file.close()





label_file_name = 'test_short.txt'
input_file_name = 'test_short.wav'
window_length = 0.04
window_step = 0.01


(X,m) = create_features(window_length, window_step, input_file_name)
y = create_labels(window_length, window_step, m, label_file_name)


(theta,hist) = train(X, y, 0, 0.001, 1500)

plt.plot(hist)
plt.ylabel('J')
plt.show()

pos = len(np.argwhere(y==1))
neg = len(np.argwhere(y==0))

print(pos)
print(neg)
print(y)

test_file_name = 'test_short.wav'
(Xtest, mtest) = create_features(window_length, window_step, test_file_name)


outfile = 'test_short_output.txt'
create_intervals(window_length, window_step, Xtest, theta, outfile, threshold = 0.6)


