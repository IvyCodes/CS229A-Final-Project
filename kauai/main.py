import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import operator
import random

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

def acc_function(y, y_test):
	num_correct = 0
	for (y_i, y_test_i) in zip(y, y_test):
		num_correct += 1 if y_i == y_test_i else 0
	return num_correct/len(y)


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

def create_features(winlen, winstep, file_name, normalized = True):
	(rate,sig) = wav.read(file_name)
	X = mfcc(sig,rate, numcep = 26, winlen = window_length, winstep = window_step)
	m = np.size(X, axis=0)
	X = (X-X.mean(axis = 0))/X.var(axis = 0)
	X = np.insert(X, 0, 1, axis = 1)
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

def split_data(X, y, test_ratio = 0.3):
	m = y.size
	mtest = np.floor(m*0.3)
	mtrain = m - mtest
	begin_i = random.randint(0,mtrain)
	end_i = int(begin_i + mtest)
	ytrain = np.append(y[0:begin_i,:], y[end_i::,:], axis = 0)
	ytest = y[begin_i:end_i,:]
	Xtrain = np.append(X[0:begin_i,:], X[end_i::,:], axis = 0)
	Xtest = X[begin_i:end_i,:]
	return (Xtrain, ytrain, Xtest, ytest)

	return (Xtrain, ytrain, Xtest, ytest)
def get_h(X, theta):
	return sigmoid(X.dot(theta))[:,0]

def smooth_h(h, window = 20):
	index_bound = len(h)
	new_result = [0]*index_bound
	for i in range(index_bound):
		lowerbd = int(max(min(i-window/2,index_bound),0))
		upperbd = int(max(min(i+window/2,index_bound),0))
		avg = np.mean(h[lowerbd:upperbd])
		new_result[i] = avg
	return new_result

def get_result(h, threshold = 0.5):
	result = []
	for i in h:
		res = 1 if i > threshold else 0
		result.append(res)
	return result

def smooth_result(result, window = 20, method = 'sliding'): #TODO: test this
	index_bound = len(result)
	if method == 'discrete':
		for i in range(int(index_bound/window) + 1):
			lowerbd = max(min(i*5,index_bound),0)
			upperbd = max(min(i*5+window,index_bound),0)
			avg = np.mean(result[lowerbd:upperbd])
			# print(avg)
			if avg > 0.5:
				result[lowerbd:upperbd] = [1]*(upperbd - lowerbd)
			else:
				result[lowerbd:upperbd] = [0]*(upperbd - lowerbd)
		return result
	elif method == 'sliding':
		new_result = [0]*len(result)
		for i in range(len(new_result)):
			lowerbd = int(max(min(i-window/2,index_bound),0))
			upperbd = int(max(min(i+window/2,index_bound),0))
			avg = np.mean(result[lowerbd:upperbd])
			if avg > 0.5:
				new_result[i] = 1
			else:
				new_result[i] = 0
		return new_result

def create_intervals(result):
	val_changes = list(map(operator.sub, result[1:-1], result[0:-2]))
	index = 0
	begin = []
	end = []
	for i in val_changes:
		if i == 1:
			begin.append(index)
		elif i == -1:
			end.append(index)
		index+=1
	if len(end) > len(begin):
		begin = [0] + begin
	elif len(begin) > len(end):
		end = end + [len(result)+1]
	return zip(begin, end)

def write_intervals(intervals, out_file, winlen):
	file = open(out_file, 'w+')
	for (x,y) in intervals:
		file.write('\t'.join((str(x*winlen),str(y*winlen),'\n')))
	file.close()



def find_lambda(X, y, min_lamb, max_lamb, step_size = 1):
	(Xtrain, ytrain, Xtest, ytest) = split_data(X,y)

	train_acc = []
	test_acc = []
	lamb = min_lamb
	(Xtrain, ytrain, Xtest, ytest) = split_data(X,y)

	while lamb < max_lamb:
		(theta,hist) = train(Xtrain, ytrain, lamb, .3, 2000)

		htest = get_h(Xtest,theta)
		smoothed_htest = smooth_h(htest)
		resulttest = get_result(smoothed_htest)
		smoothed_resulttest = smooth_result(resulttest)

		h = get_h(Xtrain,theta)
		smoothed_h = smooth_h(h)
		result = get_result(smoothed_h)
		smoothed_result = smooth_result(result)

		train_acc.append(acc_function(ytrain,smoothed_result))
		test_acc.append(acc_function(ytest,smoothed_resulttest))
		lamb += step_size
	plt.plot(train_acc, label='Training Accuracy')
	plt.plot(test_acc, label = 'Test Accuracy')
	plt.legend()
	plt.show()

def show_hist(hist):
	plt.plot(hist)
	plt.xlabel('m')
	plt.ylabel('Cost')
	plt.show()


def show_smoothing(X,y, lamb = 0, show_history = True):
	(Xtrain, ytrain, Xtest, ytest) = split_data(X,y)
	(theta,hist) = train(Xtrain, ytrain, lamb, .3, 2000)
	if show_history:
		show_hist(hist)

	htest = get_h(Xtest,theta)
	smoothed_htest = smooth_h(htest)
	resulttest = get_result(smoothed_htest)
	smoothed_resulttest = smooth_result(resulttest)

	htrain = get_h(Xtrain,theta)
	smoothed_htrain = smooth_h(htrain)
	resulttrain = get_result(smoothed_htrain)
	smoothed_resulttrain = smooth_result(resulttrain)

	plt.subplot(5, 2, 1)
	plt.plot(ytrain)
	plt.subplot(5, 2, 3)
	plt.plot(htrain)
	plt.subplot(5, 2, 5)
	plt.plot(smoothed_htrain)
	plt.subplot(5, 2, 7)
	plt.plot(resulttrain)
	plt.subplot(5, 2, 9)
	plt.plot(smoothed_resulttrain, label = str(acc_function(ytrain,smoothed_resulttrain)))
	plt.legend()

	plt.subplot(5, 2, 2)
	plt.plot(ytest)
	plt.subplot(5, 2, 4)
	plt.plot(htest)
	plt.subplot(5, 2, 6)
	plt.plot(smoothed_htest)
	plt.subplot(5, 2, 8)
	plt.plot(resulttest)
	plt.subplot(5, 2, 10)
	plt.plot(smoothed_resulttest, label = str(acc_function(ytest,smoothed_resulttest)))
	plt.legend()

	plt.show()

train_file_name = 'balanced'

window_length = 0.04
window_step = 0.01


(X,m) = create_features(window_length, window_step, train_file_name + '.wav')
y = create_labels(window_length, window_step, m, train_file_name + '.txt')

find_lambda(X, y, 0, 200, 15)


# write_intervals(create_intervals(smoothed_result), 'new_short_result.txt', window_step)














# pos = len(np.argwhere(y==1))
# neg = len(np.argwhere(y==0))

# print(pos)
# print(neg)
# print(y)


# (theta,hist) = train(X, y, 19, .35, 2000)
# (J_train, _) = cost_function(theta, X, y, 0)
# (J_test, _) = cost_function(theta, Xtest, ytest, 0)

# outfile = 'test_long_output.txt'
# create_intervals(window_length, window_step, Xtest, theta, outfile, threshold = 0.55)
# create_intervals(window_length, window_step, X, theta, 'test_short_output.txt', threshold = 0.55)


# plt.plot(hist)
# plt.xlabel('m')
# plt.ylabel('Cost')
# plt.show()



# file = open('learning_rate.txt', 'a+')


