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

def split_data(X, y, test_ratio = 0.2):
	m = y.size
	mtest = np.floor(m*test_ratio)
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

def get_result(h, threshold = 0.3):
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

def show_hist(hist):
	plt.plot(hist)
	plt.xlabel('Training Epoch')
	plt.ylabel('J')
	plt.ylim([-0.03, 1.03])
	plt.show()


def train_n_times(X, y, lamb, alpha = 0.3, num_iter = 2000, n = 1):
	train_acc = []
	test_acc = []
	thetas = np.array([[0.0]]*27)

	for i in range(n):
		(Xtrain, ytrain, Xtest, ytest) = split_data(X,y)
		(theta,hist) = train(Xtrain, ytrain, lamb, .3, 2000)
		htest = get_h(Xtest,theta)
		smoothed_htest = smooth_h(htest)
		resulttest = get_result(smoothed_htest)
		smoothed_resulttest = smooth_result(resulttest)

		h = get_h(Xtrain,theta)
		smoothed_h = smooth_h(h)
		result = get_result(smoothed_h)
		smoothed_result = smooth_result(result)
		t1 = acc_function(ytrain,smoothed_result)
		t2 = acc_function(ytest,smoothed_resulttest)
		# print(theta)
		train_acc.append(t1)
		test_acc.append(t2)
		thetas += theta
		print("train_acc = " + str(t1) + "  test_acc = " + str(t2))

	print(thetas/n)
	return (np.mean(train_acc),np.mean(test_acc),thetas/n)

def find_roc(X,y, lamb = 0.5, alpha = 0.3, num_iter = 2000):
	fpr = []
	tpr = []
	thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	(Xtrain, ytrain, Xtest, ytest) = split_data(X,y)
	(theta,hist) = train(Xtrain, ytrain, lamb, alpha, num_iter)
	htest = get_h(Xtest,theta)
	smoothed_htest = smooth_h(htest)
	for t in thresholds:
		false_positives = 0
		false_negatives = 0
		true_positives = 0
		true_negatives = 0

		resulttest = get_result(smoothed_htest, threshold = t)
		smoothed_resulttest = smooth_result(resulttest)

		for (y_i, y_test_i) in zip(ytest, smoothed_resulttest):
			if y_i == y_test_i == 1:
				true_positives += 1
			elif y_i == y_test_i == 0:
				true_negatives += 1
			elif y_i == 1 and y_test_i == 0:
				false_negatives += 1
			else:
				false_positives += 1

		fpr.append(false_positives/(false_positives + true_negatives))
		tpr.append(true_positives/(true_positives + false_negatives))

	return (tpr, fpr)



def find_lambda(X, y, min_lamb, max_lamb, step_size = 0.1):
	(Xtrain, ytrain, Xtest, ytest) = split_data(X,y)

	train_acc = []
	test_acc = []
	lambs = []
	lamb = min_lamb
	hist1 = None
	while lamb < max_lamb:
		print("training with lambda = " + str(lamb))

		(theta,hist) = train(Xtrain, ytrain, lamb, .33, 20000)
		if hist1 is None:
			hist1 = hist
		htest = get_h(Xtest,theta)
		smoothed_htest = smooth_h(htest)
		resulttest = get_result(smoothed_htest)
		smoothed_resulttest = smooth_result(resulttest)

		h = get_h(Xtrain,theta)
		smoothed_h = smooth_h(h)
		result = get_result(smoothed_h)
		smoothed_result = smooth_result(result)

		t1 = acc_function(ytrain,smoothed_result)
		t2 = acc_function(ytest,smoothed_resulttest)
		# print(theta)
		train_acc.append(1-t1)
		test_acc.append(1-t2)
		lambs.append(lamb)


		# (a1, a2, theta) = train_n_times(X, y, lamb, alpha = 0.3, num_iter = 2000, n = 15)
		# train_acc.append(a1)
		# test_acc.append(a2)

		lamb += step_size
	show_hist(hist1)
	plt.plot(lambs, train_acc,label='Training Error')
	plt.plot(lambs, test_acc,label = 'Validation Error')
	plt.xlabel('Lambda')
	plt.ylabel('Error')
	plt.ylim([-0.03, 1.03])
	plt.legend()
	plt.show()


def show_smoothing(X,y, lamb = 0, show_history = True, theta = None):
	(bottom,top) = (3000,5000)
	form = 1 #makes pictures look nice
	if theta is None:
		form = 0
		(Xtrain, ytrain, X, y) = split_data(X,y)
		(theta,hist) = train(Xtrain, ytrain, lamb, .2, 2000)
		if show_history:
			show_hist(hist)
		htrain = get_h(Xtrain,theta)
		smoothed_htrain = smooth_h(htrain)
		resulttrain = get_result(smoothed_htrain)
		smoothed_resulttrain = smooth_result(resulttrain)
		plt.subplot(5, 2, 1)
		plt.plot(ytrain)
		plt.title("Labeled Data", fontdict={'fontsize': 10})
		plt.xlim([bottom,top])
		plt.ylim([-0.03, 1.03])

		plt.subplot(5, 2, 3)
		plt.plot(htrain)
		plt.title("Hypothesis", fontdict={'fontsize': 10})
		plt.xlim([bottom,top])
		plt.ylim([-0.03, 1.03])

		plt.subplot(5, 2, 5)
		plt.plot(smoothed_htrain)
		plt.title("Smoothed Hypothesis", fontdict={'fontsize': 10})
		plt.xlim([bottom,top])
		plt.ylim([-0.03, 1.03])

		plt.subplot(5, 2, 7)
		plt.plot(resulttrain)
		plt.title("Prediction", fontdict={'fontsize': 10})
		plt.xlim([bottom,top])
		plt.ylim([-0.03, 1.03])

		plt.subplot(5, 2, 9)
		plt.title("Smoothed Result" + " (Accuracy = " + str(np.around(acc_function(ytrain,smoothed_resulttrain), decimals = 5)) + ")", fontdict={'fontsize': 10})
		plt.plot(smoothed_resulttrain)
		plt.xlim([bottom,top])
		plt.ylim([-0.03, 1.03])

	htest = get_h(X,theta)
	smoothed_htest = smooth_h(htest)
	resulttest = get_result(smoothed_htest)
	smoothed_resulttest = smooth_result(resulttest)

	plt.subplot(5, 2-form, 2/(form+1))
	plt.plot(y)
	plt.title("Labeled Data", fontdict={'fontsize': 10})
	plt.xlim([bottom,top])
	plt.ylim([-0.03, 1.03])
	

	plt.subplot(5, 2-form, 4/(form+1))
	plt.plot(htest)
	plt.title("Hypothesis", fontdict={'fontsize': 10})
	plt.xlim([bottom,top])
	plt.ylim([-0.03, 1.03])
	plt.tight_layout()

	plt.subplot(5, 2-form, 6/(form+1))
	plt.plot(smoothed_htest)
	plt.title("Smoothed Hypothesis", fontdict={'fontsize': 10})
	plt.xlim([bottom,top])
	plt.ylim([-0.03, 1.03])
	plt.tight_layout()

	plt.subplot(5, 2-form, 8/(form+1))
	plt.plot(resulttest)
	plt.title("Result", fontdict={'fontsize': 10})
	plt.xlim([bottom,top])
	plt.ylim([-0.03, 1.03])
	plt.tight_layout()

	plt.subplot(5, 2-form, 10/(form+1))
	plt.title("Smoothed Result" + " (Accuracy = " + str(np.around(acc_function(y,smoothed_resulttest), decimals = 5)) + ")", fontdict={'fontsize': 10})
	plt.plot(smoothed_resulttest)
	plt.xlim([bottom,top])
	plt.ylim([-0.03, 1.03])
	plt.tight_layout()

	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.6)
	plt.show()

train_file_name = 'new_data'
test_file_name = 'imbalanced'

window_length = 0.025
window_step = 0.01


(X,m) = create_features(window_length, window_step, train_file_name + '.wav')
y = create_labels(window_length, window_step, m, train_file_name + '.txt')

# (tpr, fpr) = find_roc(X,y, lamb = 0.5, alpha = 0.3, num_iter = 2000)
# print(tpr)
# print(fpr)

# (Xtest,m) = create_features(window_length, window_step, train_file_name + '.wav')
# ytest = create_labels(window_length, window_step, m, train_file_name + '.txt')

# show_smoothing(X,y, lamb = 0, show_history = False, theta = None)

find_lambda(X, y, 0, .2, .02)

# (a1, a2, theta) = train_n_times(X, y, lamb=0, alpha = 0.3, num_iter = 2000, n = 1)
# show_smoothing(Xtest,ytest, lamb = 0.5, show_history = False, theta = theta)




