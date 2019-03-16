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

def acc_function(y, y_test):
	for 

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

def create_intervals(winlen, winstep, X, theta, out_file_name, threshold = 0.5, smooth_step = 10, ignore_step = 30, confidence_threshold = .66):
	h = sigmoid(X.dot(theta))
	print('number of windows: ' + str(h.size))

	result = np.argwhere(h > threshold)[:,0]
	# print(result)
	intervals = []
	begin_i = result[0]
	end_i = result[0]
	confidence_vals = []
	for i in result:
		if i <= end_i + smooth_step:
			end_i = i
			confidence_vals.append(h[i,0])
		else:
			if end_i-begin_i > ignore_step:
				confidence = np.around(np.mean(confidence_vals), decimals = 2)
				if confidence > confidence_threshold:
					begin_time = begin_i*winstep
					end_time = end_i * winstep + winlen
					intervals.append([str(begin_time), str(end_time), str(confidence)+'\n'])
			begin_i = i
			end_i = i
			confidence_vals = []

	file = open(out_file_name, 'w+')
	for i in intervals:
		out_string = "\t".join(i)
		#print(i)
		file.write(out_string)
	file.close()





train_file_name = 'test_short'
test_file_name = 'test_long'

window_length = 0.04
window_step = 0.01


(X,m) = create_features(window_length, window_step, train_file_name + '.wav')
y = create_labels(window_length, window_step, m, train_file_name + '.txt')

(Xtest, mtest) = create_features(window_length, window_step, test_file_name + '.wav')
ytest = create_labels(window_length, window_step, mtest, test_file_name + '.txt')

lambs = []
train_errs = []
test_errs = []
min_lamb = 0
min_err = 10
for lamb in range(0,300):
	# file.write(str(lamb) + "\t" + str(J_train) + "\t" + str(J_test) + "\n")
	(theta,hist) = train(X, y, lamb, .35, 2000)
	(J_train, _) = cost_function(theta, X, y, 0)
	(J_test, _) = cost_function(theta, Xtest, ytest, 0)

	lambs.append(lamb)
	train_errs.append(J_train[0,0])
	test_errs.append(J_test[0,0])

	if J_test[0,0] < min_err:
		min_lamb = lamb
		min_err = J_test[0,0]

	# print("training error = " + str(J_train))
	# print("test error = " + str(J_test))

	if lamb == 200:
		outfile = 'test_long_output.txt'
		create_intervals(window_length, window_step, Xtest, theta, outfile, threshold = 0.55)
		create_intervals(window_length, window_step, X, theta, 'test_short_output.txt', threshold = 0.55)

print(min_lamb)

plt.plot(lambs, test_errs, 'r', label='Validation error')
plt.plot(lambs, train_errs, 'b', label = 'Training error')
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.title('Training error')
plt.legend()
plt.show()

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


