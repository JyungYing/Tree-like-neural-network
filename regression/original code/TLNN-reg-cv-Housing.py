#=======================================
## Tree-like neural network - Housing (Cross_validation)
#=======================================

###################

# Decision tree: R codes: Housing Compressive Strength dataset.

#----------------------------------------
# Load the Housing Compressive Strength dataset.

data_original = read.table("D:/Li-Chun-Ying/Data-Sets/regression/Housing-Training-Data.txt", header = FALSE) 

str(data_original)

#----------------------------------------
# regression tree using rpart()

library(rpart)

housing_rpart = rpart(V14 ~ ., data = data_original, 
	control = rpart.control(xval = 10, maxdepth = 3))

library(partykit) 

plot(as.party(housing_rpart), tp_args = list(id = FALSE)) 

as.party(housing_rpart) 

#----------------------------------------
# regression using ctree()

library("party")

housing_ctree = ctree(V14 ~ ., data = data_original, controls = ctree_control(maxdepth = 3))

plot(housing_ctree)
windows()

plot(housing_ctree, type = 'simple')

housing_ctree

###################  

# Neural Network: Python codes: Housing

#----------------------------------------
# Change the current working directory to the specified path.

import os

mywd = "C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Data-Sets\\regression"
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
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, concatenate, Dropout
#from keras.layers import add, subtract, multiply, average, maximum, dot, Lambda

#----------------------------------------
# Load the Housing Compressive Strength dataset.

dataset = np.loadtxt("Housing-Training-Data.txt")

type(dataset)

dataset.shape

X = dataset[:, 0:13]
Y = dataset[:, 13]

X.shape
Y.shape

# Input variable of tree_like_nn

x_group = [[5], [12], [0], [7], [5], [12]]

#for i in range(len(x_group)):
#    locals()['X%s'%(i + 1)] = dataset[:, x_group[i]]

X1 = dataset[:, x_group[0]]
X2 = dataset[:, x_group[1]]
X3 = dataset[:, x_group[2]]
X4 = dataset[:, x_group[3]]
X5 = dataset[:, x_group[4]]
X6 = dataset[:, x_group[5]]

X1.shape; X2.shape; X3.shape; X4.shape; X5.shape; X6.shape
Y.shape

#----------------------------------------
# Standardize the input and output data.

X_sample_mean = np.mean(X, axis = 0)
np.round(X_sample_mean, 4)

X_sample_std = np.std(X, axis = 0, ddof = 1)
np.round(X_sample_std, 4)

#for i in range(len(x_group)):
#    locals()['X%s'%(i + 1) + '_sample_mean'] = X_sample_mean[x_group[i]]
#    print('X%s'%(i + 1) + '_sample_mean = ', np.round(X_sample_mean[x_group[i]], 4))
#    locals()['X%s'%(i + 1) + '_sample_std'] = X_sample_std[x_group[i]]
#    print('X%s'%(i + 1) + '_sample_std = ', np.round(X_sample_std[x_group[i]], 4))

X1_sample_mean = X_sample_mean[x_group[0]]
np.round(X1_sample_mean, 4)

X1_sample_std = X_sample_std[x_group[0]]
np.round(X1_sample_std, 4)

X2_sample_mean = X_sample_mean[x_group[1]]
np.round(X2_sample_mean, 4)

X2_sample_std = X_sample_std[x_group[1]]
np.round(X2_sample_std, 4)

X3_sample_mean = X_sample_mean[x_group[2]]
np.round(X3_sample_mean, 4)

X3_sample_std = X_sample_std[x_group[2]]
np.round(X3_sample_std, 4)

X4_sample_mean = X_sample_mean[x_group[3]]
np.round(X4_sample_mean, 4)

X4_sample_std = X_sample_std[x_group[3]]
np.round(X4_sample_std, 4)

X5_sample_mean = X_sample_mean[x_group[4]]
np.round(X5_sample_mean, 4)

X5_sample_std = X_sample_std[x_group[4]]
np.round(X5_sample_std, 4)

X6_sample_mean = X_sample_mean[x_group[5]]
np.round(X6_sample_mean, 4)

X6_sample_std = X_sample_std[x_group[5]]
np.round(X6_sample_std, 4)

Y_sample_mean = np.mean(Y, axis = 0)
np.round(Y_sample_mean, 4)

Y_sample_std = np.std(Y, axis = 0, ddof = 1)
np.round(Y_sample_std, 4)

#----------------------------------------
standardize = lambda x: (x - np.mean(x, axis = 0)) / np.std(x, axis = 0, ddof = 1)

X_train = standardize(X)

#trains = []
#for i in range(len(x_group)):
#    locals()['X%s'%(i + 1) + '_train'] = X_train[:, x_group[i]]
#    trains = trains + [locals()['X%s'%(i + 1) + '_train']]

X1_train = X_train[:, x_group[0]]
X2_train = X_train[:, x_group[1]]
X3_train = X_train[:, x_group[2]]
X4_train = X_train[:, x_group[3]]
X5_train = X_train[:, x_group[4]]
X6_train = X_train[:, x_group[5]]

np.round(np.mean(X1_train, axis = 0), 4)
np.round(np.std(X1_train, axis = 0, ddof = 1), 4)

np.round(np.mean(X2_train, axis = 0), 4)
np.round(np.std(X2_train, axis = 0, ddof = 1), 4)

np.round(np.mean(X3_train, axis = 0), 4)
np.round(np.std(X3_train, axis = 0, ddof = 1), 4)

np.round(np.mean(X4_train, axis = 0), 4)
np.round(np.std(X4_train, axis = 0, ddof = 1), 4)

np.round(np.mean(X5_train, axis = 0), 4)
np.round(np.std(X5_train, axis = 0, ddof = 1), 4)

np.round(np.mean(X6_train, axis = 0), 4)
np.round(np.std(X6_train, axis = 0, ddof = 1), 4)

Y_train = standardize(Y)

np.round(np.mean(Y_train, axis = 0), 4)
np.round(np.std(Y_train, axis = 0, ddof = 1), 4)

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
# 10-fold cross validation

cv_mse = []
cv_mae = []

for k in range(fold_num):
  input_g1 = Input(shape = (1,))
  x1 = input_g1
  t1 = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x1)
  t1 = Dropout(rate = 0.1)(t1)
  #
  y11 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t1)
  y12 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t1)
  #
  input_g2 = Input(shape = (1,))
  x2 = concatenate([y11, input_g2], axis = 1)  # column bind
  t2 = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x2)
  t2 = Dropout(rate = 0.1)(t2)
  #
  y21 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t2)
  y22 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t2)
  #
  input_g3 = Input(shape = (1,))
  x3 = concatenate([y21, input_g3], axis = 1)  # column bind
  t3 = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x3)
  t3 = Dropout(rate = 0.1)(t3)
  #
  y31 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t3)
  y32 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t3)
  #
  input_g6 = Input(shape = (1,))
  x6 = concatenate([y22, input_g6], axis = 1)  # column bind
  t6 = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x6)
  t6 = Dropout(rate = 0.1)(t6)
  #
  y61 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t6)
  y62 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t6)
  #
  input_g9 = Input(shape = (1,))
  x9 = concatenate([y12, input_g9], axis = 1)  # column bind
  t9 = Dense(units = 3, kernel_initializer = "uniform", activation = "linear")(x9)
  t9 = Dropout(rate = 0.1)(t9)
  #
  y91 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t9)
  y92 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t9)
  #
  input_g10 = Input(shape = (1,))
  x10 = concatenate([y91, input_g10], axis = 1)  # column bind
  t10 = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x10)
  t10 = Dropout(rate = 0.1)(t10)
  #
  y101 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t10)
  y102 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t10)
  #
  x_final =  concatenate([y31, y32, y61, y62, y92, y101, y102], axis = 1)  # column bind
  y_final = Dense(units = 5, kernel_initializer = "uniform", activation = "relu")(x_final)
  y_final = Dropout(rate = 0.1)(y_final)
  y_final = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(y_final)
  response = y_final
  #
  model = Model(inputs = [input_g1, input_g2, input_g3, input_g6, input_g9, input_g10], 
		outputs = response)
  #
  #model.summary()
  # Compile the model.
  model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
  # Select index sets.
  seq_index = range(index_begin[k], (index_begin[k] + fold_length[k]))
  test_index = index[seq_index]
  train_index = np.delete(index, seq_index)
  # Fit the model.
  history = model.fit([X1_train[train_index], X2_train[train_index], X3_train[train_index]
  , X4_train[train_index], X5_train[train_index], X6_train[train_index]]
  , Y_train[train_index], validation_split = 0.2, epochs = 1000, batch_size = 20, verbose = 2)
  # Evaluate the model.
  scores = model.evaluate([X1_train[test_index], X2_train[test_index], X3_train[test_index]
  , X4_train[test_index], X5_train[test_index], X6_train[test_index]]
  , Y_train[test_index], verbose = 0)
  print("%s: %.4f" % (model.metrics_names[0], scores[0]))
  print("%s: %.4f" % (model.metrics_names[1], scores[1]))
  cv_mse.append(scores[0])
  cv_mae.append(scores[1])

print("%.4f (+/- %.4f)" % (np.mean(cv_mse), np.std(cv_mse)))
print("%.4f (+/- %.4f)" % (np.median(cv_mse), np.std(cv_mse)))

print("%.4f (+/- %.4f)" % (np.mean(cv_mae), np.std(cv_mae)))
print("%.4f (+/- %.4f)" % (np.median(cv_mae), np.std(cv_mae)))

result = {'history': history.history, 'loss_metric': loss_metric}

#----------------------------------------
# plot the model

from keras.utils import plot_model

plot_model(model, to_file = 'TLNN_housing_model0.png')
plot_model(model, show_shapes = True, to_file = 'TLNN_housing_model1.png')

#----------------------------------------
# Save the result to NPZ file

np.savez('D:\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Tree_like_nn_cv_result(Housing)', **result)

# Load the result of NPZ file

result = np.load('D:\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Tree_like_nn_cv_result(Housing).npz')

result.files

result['history']
result['loss_metric']

#----------------------------------------
# Save the model to H5 file

file_path_h5 = 'D:\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Tree_like_nn_cv(Housing).h5'
model.save(file_path_h5)

# Load model of H5 file

file_path_h5 = 'D:\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Regression\\Tree_like_nn_cv(Housing).h5'
loaded_model = load_model(file_path_h5)

#----------------------------------------
# List all data in history.

history.history.keys()

#----------------------------------------
# Summarize history for loss.

plt.ion()

plt.figure('TLNN Housing loss', figsize = (4.8, 4.0))
plt.plot(history.history['loss'], "r-")
plt.plot(history.history['val_loss'], "b--")
plt.title('TLNN Housing Training/validating loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validating loss'], loc = "best", frameon = False)
#plt.show()

#----------------------------------------
# Summarize history for metric.

plt.figure('TLNN Housing metric', figsize = (4.8, 4.0))
plt.plot(history.history['mean_absolute_error'], "r-")
plt.plot(history.history['val_mean_absolute_error'], "b--")
plt.title('TLNN Housing Training/validating metric')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['training metric', 'validating metric'], loc = "best", frameon = False)
#plt.show()

#----------------------------------------
# Plot the observed and fitted response values.

fitted = model.predict([X1_train, X2_train, X3_train, X4_train, X5_train, X6_train])  # Here, this is used to compute fitted response values.
fitted = fitted[:, 0]

fitted.shape
Y_train.shape

all = np.concatenate((Y_train, fitted))
draw_min = np.floor(np.min(all))
draw_max = np.ceil(np.max(all))

plt.figure('TLNN Housing', figsize = (4.8, 4.0))
#plt.plot(Y_train, fitted, "wo", markersize = 2, markeredgecolor = "black")
plt.plot(fitted, Y_train, 'wo', markersize = 2, markeredgecolor = "black")
plt.plot([draw_min, draw_max], [draw_min, draw_max], '-', linewidth = 1, color = 'red')
plt.title('TLNN Housing Observed versus fitted values')
plt.ylabel('observed values')
plt.xlabel('fitted values')
#plt.show()
