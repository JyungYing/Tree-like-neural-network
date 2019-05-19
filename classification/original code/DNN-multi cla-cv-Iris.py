#=======================================
## DNN - Iris
#=======================================

# Need compile Create_DNN_model.

#----------------------------------------
# Change the current working directory to the specified path.

import os

mywd = "C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Data-Sets\\classification"
os.chdir(mywd)
os.getcwd()

#----------------------------------------
# Set random seed for reproducibility.

seed = 543
from numpy import random
random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)

# Import classes and functions.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

#----------------------------------------
# Load the iris dataset.

dataset = np.loadtxt("IRIS-Training-Data.txt")

print(type(dataset))

print(dataset.shape)

index = [2, 3]

X = dataset[:, index]
Y = dataset[:, 4]

print(X.shape)
print(Y.shape)

Y = np.array(Y, dtype = int)

print(np.unique(Y))

#----------------------------------------
# one-hot encoding of output variable.

Y_train = np.zeros((len(Y), 3), dtype = int)

for i in range(len(Y)):
	Y_train[i, Y[i] - 1] = 1

print(np.unique(Y_train))

#----------------------------------------
# Standardize the input data.

X_sample_mean = np.mean(X, axis = 0)
print(np.round(X_sample_mean, 4))

X_sample_std = np.std(X, axis = 0, ddof = 1)
print(np.round(X_sample_std, 4))

standardize = lambda x: (x - np.mean(x, axis = 0)) / np.std(x, axis = 0, ddof = 1)

X_train = standardize(X)

print(np.round(np.mean(X_train, axis = 0), 4))
print(np.round(np.std(X_train, axis = 0, ddof = 1), 4))

#----------------------------------------
# Define 10-fold cross validation test index sets.

fold_num = 10

L = len(X_train)

quotient = L // fold_num
remainder = L % fold_num

fold_length = np.array([quotient]*fold_num)
if remainder > 0:
    fold_length[0:remainder] = fold_length[0:remainder] + 1

index_begin = np.zeros(fold_num, dtype = int)
index_begin[0] = 0

for j in range(1, fold_num):
    index_begin[j] = index_begin[j-1] + fold_length[j-1]

index = np.random.choice(np.arange(L), size = L, replace = False)

#----------------------------------------
# Create the DNN_model

input_dim = X.shape[1]

nodes = [input_dim, 6, 3, 3]
	# first component: number of input nodes
	# last component: number of output nodes

drop = [0, 0.2, 0.2]
	# len(dropout_rate) = len(node) - 1
	# first component: dropout rate for the input layer
	# other components: dropout rates for the hidden layers
	# Dropout rate is not used in the output layer.

#dropout_rate = [0]*(len(node)-1)
#dropout_rate = [0] + [0.2]*(len(node)-2)

kernel = ['uniform', 'uniform', 'uniform']

act = ['relu', 'relu', 'softmax']

#----------------------------------------
# 10-fold cross validation

cv_cc = []
cv_acc = []

for k in range(fold_num):
    # Build the model.
    model = create_DNN_model(nodes, drop, kernel, act)
    # Compile the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # Select index sets.
    seq_index = range(index_begin[k], (index_begin[k] + fold_length[k]))
    test_index = index[seq_index]
    train_index = np.delete(index, seq_index)
    # Fit the model.
    history = model.fit(X_train[train_index], Y_train[train_index], validation_split = 0.2, epochs = 1000, batch_size = 20, verbose = 1)
    # Evaluate the model.
    scores = model.evaluate(X_train[test_index], Y_train[test_index], verbose = 1)
    print("%s: %.4f" % (model.metrics_names[0], scores[0]))
    print("%s: %.4f" % (model.metrics_names[1], scores[1]))
    cv_cc.append(scores[0])
    cv_acc.append(scores[1])

model.summary()	

print("%.4f (+/- %.4f)" % (np.mean(cv_cc), np.std(cv_cc)))
print("%.4f (+/- %.4f)" % (np.median(cv_cc), np.std(cv_cc)))

print("%.4f (+/- %.4f)" % (np.mean(cv_acc), np.std(cv_acc)))
print("%.4f (+/- %.4f)" % (np.median(cv_acc), np.std(cv_acc)))

#----------------------------------------
# Predict

model.predict(X_train, verbose = 1)

result = {'history': history.history, 'loss_metric': scores}

#----------------------------------------
# Save the result to NPZ file

np.savez('C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\DNN_cv_result(Iris)', **result)

# Load the result of NPZ file

result = np.load('C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\DNN_cv_result(Iris).npz')

result.files

result['history']
result['loss_metric']

from keras.utils import plot_model

plot_model(model, to_file = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Figures\\Tree-like\\Classification\\Iris\\DNN_Iris_model0.png')
plot_model(model, show_shapes = True, to_file = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Figures\\Tree-like\\Classification\\Iris\\DNN_Iris_model1.png')


#----------------------------------------
# Save the model to H5 file

file_path_hdf5 = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\DNN_cv(Iris).h5'
model.save(file_path_hdf5)

# Load model of H5 file

file_path_hdf5 = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\DNN_cv(Iris).h5'
loaded_model = load_model(file_path_hdf5)

#----------------------------------------
# List all data in history.

history.history.keys()

#----------------------------------------
# Summarize history for loss.

plt.figure('DNN Iris loss', figsize = (4.8, 4.0))
plt.plot(history.history['loss'], "r-")
plt.plot(history.history['val_loss'], "b--")
plt.title('DNN Iris' + '\n' + 'Training/validating loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validating loss'], loc = "best", frameon = False)
plt.show()

#----------------------------------------
# Summarize history for metric.

plt.figure('DNN Iris metric', figsize = (4.8, 4.0))
plt.plot(history.history['acc'], "r-")
plt.plot(history.history['val_acc'], "b--")
plt.title('DNN Iris' + '\n' + 'Training/validating metric')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['training metric', 'validating metric'], loc = "best", frameon = False)
plt.show()

###################
