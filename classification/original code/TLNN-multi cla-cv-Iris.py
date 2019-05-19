#=======================================
## Tree-like neural network - Iris
#=======================================

###################

# Decision tree: R codes: Iris dataset.

#----------------------------------------
# Load the Iris dataset.

data_original = read.table("C:/Users/jghsieh/Desktop/Li-Chun-Ying/Data-Sets/classification/IRIS-Training-Data.txt", header = FALSE)

data_original[,5] = factor(data_original[,5])

str(data_original)

#----------------------------------------
# classification tree using rpart()

library(rpart)

iris_rpart = rpart(V5 ~ ., data = data_original, control = rpart.control(xval = 10, maxdepth = 3))

library(partykit) 

plot(as.party(iris_rpart), tp_args = list(id = FALSE))  

as.party(iris_rpart)

#----------------------------------------
# classification using ctree()

library("party")

iris_ctree = ctree(V5 ~ ., data = data_original, controls = ctree_control(maxdepth = 3))

plot(iris_ctree)
windows()
plot(iris_ctree, type = 'simple')

iris_ctree

###################

# Neural Network：Python codes：Iris

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
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate, Dropout

#----------------------------------------
# Load the iris dataset.

dataset = np.loadtxt("IRIS-Training-Data.txt")

print(type(dataset))

print(dataset.shape)

x_group = [[2], [3]]

X = dataset[:, 0:4]
Y = dataset[:, 4]

print(X.shape)
print(Y.shape)

X1 = dataset[:, x_group[0]]
X2 = dataset[:, x_group[1]]

print(X1.shape)
print(X2.shape)
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
# Standardize the input and output data.

X_sample_mean = np.mean(X, axis = 0)
print(np.round(X_sample_mean, 4))

X_sample_std = np.std(X, axis = 0, ddof = 1)
print(np.round(X_sample_std, 4))

X1_sample_mean = X_sample_mean[x_group[0]]
print(np.round(X1_sample_mean, 4))

X1_sample_std = X_sample_std[x_group[0]]
print(np.round(X1_sample_std, 4))

X2_sample_mean = X_sample_mean[x_group[1]]
print(np.round(X2_sample_mean, 4))

X2_sample_std = X_sample_std[x_group[1]]
print(np.round(X2_sample_std, 4))

standardize = lambda x: (x - np.mean(x, axis = 0)) / np.std(x, axis = 0, ddof = 1)

X_train = standardize(X)

X1_train = X_train[:, x_group[0]]
X2_train = X_train[:, x_group[1]]

print(np.round(np.mean(X1_train, axis = 0), 4))
print(np.round(np.std(X1_train, axis = 0, ddof = 1), 4))

print(np.round(np.mean(X2_train, axis = 0), 4))
print(np.round(np.std(X2_train, axis = 0, ddof = 1), 4))

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

cv_cc = []
cv_acc = []

for k in range(fold_num):
    # Build the tree like nn model
    input_g1 = Input(shape = (1,))
    x1 = input_g1
    t1 = Dense(units = 2, kernel_initializer = "uniform", activation = "relu")(x1)
    t1 = Dropout(rate = 0.2)(t1)
    #
    y11 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t1)
    y12 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t1)
    #
    input_g3 = Input(shape = (1,))
    x3 = concatenate([y12, input_g3], axis = 1)  # column bind
    t3 = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x3)
    t3 = Dropout(rate = 0.2)(t3)
    #
    y31 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t3)
    y32 = Dense(units = 1, kernel_initializer = "uniform", activation = "linear")(t3)
    #
    x_final =  concatenate([y11, y31, y32], axis = 1)  # column bind
    y_final = Dense(units = 3, kernel_initializer = "uniform", activation = "relu")(x_final)
    y_final = Dropout(rate = 0.2)(y_final)
    y_final = Dense(units = 3, kernel_initializer = "uniform", activation = "softmax")(y_final)
    response = y_final
    #
    model = Model(inputs = [input_g1, input_g3], outputs = response)
    # Compile the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # Select index sets.
    seq_index = range(index_begin[k], (index_begin[k] + fold_length[k]))
    test_index = index[seq_index]
    train_index = np.delete(index, seq_index)
    # Fit the model.
    history = model.fit([X1_train[train_index], X2_train[train_index]]
                        , Y_train[train_index], validation_split = 0.2, epochs = 1000, batch_size = 20, verbose = 1)
    # Evaluate the model.
    scores = model.evaluate([X1_train[test_index], X2_train[test_index]]
                            , Y_train[test_index], verbose = 1)
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

model.predict([X1_train, X2_train], verbose = 1)

result = {'history': history.history, 'loss_metric': scores}

#----------------------------------------
# Save the result to NPZ file

np.savez('C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\Tree_like_nn_cv_result(Iris)', **result)

# Load the result of NPZ file

result = np.load('C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\Tree_like_nn_cv_result(Iris).npz')

result.files

result['history']
result['scores']

#----------------------------------------
# plot the model.

from keras.utils import plot_model

plot_model(model, to_file = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Figures\\Tree-like\\Classification\\Iris\\TLNN_Iris_model0.png')
plot_model(model, show_shapes = True, to_file = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Figures\\Tree-like\\Classification\\Iris\\TLNN_Iris_model1.png')

#----------------------------------------
# Save the model to H5 file

file_path_hdf5 = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\Tree_like_nn_cv(Iris).h5'
model.save(file_path_hdf5)

# Load model of H5 file

file_path_hdf5 = 'C:\\Users\\jghsieh\\Desktop\\Li-Chun-Ying\\Keras-Objects\\tree-like-nn\\Classification\\Iris\\Tree_like_nn_cv(Iris).h5'
loaded_model = load_model(file_path_hdf5)

#----------------------------------------
# List all data in history.

history.history.keys()

#----------------------------------------
# Summarize history for loss.

plt.figure('TLNN Iris loss', figsize = (4.8, 4.0))
plt.plot(history.history['loss'], "r-")
plt.plot(history.history['val_loss'], "b--")
plt.title('TLNN Iris' + '\n' + 'Training/validating loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validating loss'], loc = "best", frameon = False)
plt.show()

#----------------------------------------
# Summarize history for metric.

plt.figure('TLNN Iris metric', figsize = (4.8, 4.0))
plt.plot(history.history['acc'], "r-")
plt.plot(history.history['val_acc'], "b--")
plt.title('TLNN Iris' + '\n' + 'Training/validating metric')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['training metric', 'validating metric'], loc = "best", frameon = False)
plt.show()

###################
