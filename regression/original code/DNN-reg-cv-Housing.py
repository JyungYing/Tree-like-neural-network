#=======================================
## DNN - Housing
#=======================================

# Need compile Create_DNN_model.

#----------------------------------------
# Change the working directory.

import os

mywd = "C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Data-Sets\\regression"
os.chdir(mywd)
os.getcwd()

#----------------------------------------
# Set random seed.

seed = 543
from numpy import random
random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)

#----------------------------------------
# Import classes and functions.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

#----------------------------------------
# Load the dataset.

dataset = np.loadtxt("Housing-Training-Data.txt")

print(dataset.shape)

index = [0, 5, 7, 12]

X = dataset[:, index]
Y = dataset[:, 13]

#----------------------------------------
# Standardize the input and output data.

X_sample_mean = np.mean(X, axis = 0)
print(np.round(X_sample_mean, 4))

X_sample_std = np.std(X, axis = 0, ddof = 1)
print(np.round(X_sample_std, 4))

Y_sample_mean = np.mean(Y, axis = 0)
print(np.round(Y_sample_mean, 4))

Y_sample_std = np.std(Y, axis = 0, ddof = 1)
print(np.round(Y_sample_std, 4))

standardize = lambda x:(x - np.mean(x, axis = 0))/np.std(x, axis = 0, ddof = 1)

X_train = standardize(X)
Y_train = standardize(Y)

print(np.round(np.mean(X_train, axis = 0), 4))
print(np.round(np.std(X_train, axis = 0, ddof = 1), 4))

print(np.round(np.mean(Y_train, axis = 0), 4))
print(np.round(np.std(Y_train, axis = 0, ddof = 1), 4))

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

nodes = [input_dim, 7, 7, 6, 1]
	# first component: number of input nodes
	# last component: number of output nodes

drop = [0, 0.2, 0.2, 0.2]
	# len(dropout_rate) = len(node) - 1
	# first component: dropout rate for the input layer
	# other components: dropout rates for the hidden layers
	# Dropout rate is not used in the output layer.

#dropout_rate = [0]*(len(node)-1)
#dropout_rate = [0] + [0.2]*(len(node)-2)

kernel = ['uniform', 'uniform', 'uniform', 'uniform']

act = ['relu', 'relu', 'relu', 'linear']

#----------------------------------------
# 10-fold cross validation

cv_mse = []
cv_mae = []

for k in range(fold_num):
    # Create the DNN_model.
    model = create_DNN_model(nodes, drop, kernel, act)
    # Compile the model.
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mae'])
    # Select index sets.
    seq_index = range(index_begin[k], (index_begin[k] + fold_length[k]))
    test_index = index[seq_index]
    train_index = np.delete(index, seq_index)
    # Fit the model.
    history = model.fit(X_train[train_index], Y_train[train_index], epochs = 1000, validation_split = 0.2, batch_size = 20, verbose = 1)
    # Evaluate the model.
    scores = model.evaluate(X_train[test_index], Y_train[test_index], verbose = 1)
    print("%s: %.4f" % (model.metrics_names[0], scores[0]))
    print("%s: %.4f" % (model.metrics_names[1], scores[1]))
    cv_mse.append(scores[0])
    cv_mae.append(scores[1])

model.summary()

print("%.4f (+/- %.4f)" % (np.mean(cv_mse), np.std(cv_mse)))
print("%.4f (+/- %.4f)" % (np.median(cv_mse), np.std(cv_mse)))

print("%.4f (+/- %.4f)" % (np.mean(cv_mae), np.std(cv_mae)))
print("%.4f (+/- %.4f)" % (np.median(cv_mae), np.std(cv_mae)))

#----------------------------------------
# Predict

model.predict(X_train, verbose = 1)

result = {'history': history.history, 'loss_metric': scores}

#----------------------------------------
# Save the result to NPZ file

np.savez('C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Housing\\DNN_cv_result(Housing)', **result)

# Load the result of NPZ file

result = np.load('C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Housing\\DNN_cv_result(Housing).npz')

result.files

result['history']
result['loss_metric']

#----------------------------------------
# plot the model

from keras.utils import plot_model

plot_model(model, to_file = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Figures\\Tree-like\\Regression\\Housing\\DNN_housing_model0.png')
plot_model(model, show_shapes = True, to_file = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Figures\\Tree-like\\Regression\\Housing\\DNN_housing_model1.png')

#----------------------------------------
# Save the model to HDF5 file

file_path_hdf5 = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Housing\\DNN_cv(Housing)'
model.save(file_path_hdf5)

# Load model of HDF5 file

file_path_hdf5 = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Housing\\DNN_cv(Housing)'
laoded_model = load_model(file_path_hdf5)

#----------------------------------------
# List all data in history.

history.history.keys()

#----------------------------------------
# Summarize history for loss.

plt.figure('DNN Housing loss', figsize = (4.8, 4.0))
plt.plot(history.history['loss'], "r-")
plt.plot(history.history['val_loss'], "b--")
plt.title('DNN Housing' + '\n' + 'Training/validating loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validating loss'], loc = "best", frameon = False)
plt.show()

#----------------------------------------
# Summarize history for metric.

plt.figure('DNN Housing metric', figsize = (4.8, 4.0))
plt.plot(history.history['mean_absolute_error'], "r-")
plt.plot(history.history['val_mean_absolute_error'], "b--")
plt.title('DNN Housing' + '\n' + 'Training/validating metric')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['training metric', 'validating metric'], loc = "best", frameon = False)
plt.show()

#----------------------------------------
# Make prediction.

fitted = model.predict(X_train)
fitted = fitted[:, 0]

print(fitted.shape)

# Plot

all = np.concatenate((Y_train, fitted))
draw_min = np.floor(np.min(all))
draw_max = np.ceil(np.max(all))

plt.figure('DNN_Housing', figsize = (4.8, 4.0))
plt.plot(fitted, Y_train, 'wo', markersize = 2, markeredgecolor = "black")
plt.plot([draw_min, draw_max], [draw_min, draw_max], '-', linewidth = 1, color = 'red')
plt.title('DNN Housing' + '\n' + 'Observed versus fitted values')
plt.ylabel('observed values')
plt.xlabel('fitted values')
plt.show()
