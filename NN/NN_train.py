from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def load_data():
    num_cols = 56 * 4 + 3
    data = np.genfromtxt("../generacion_datos/datos.txt")
    print("data_len",len(data))
    data = np.unique(data,axis=0)
    print("data_len",len(data))
    X = data[:, :num_cols - 3]
    Y = data[:, -3:]
    return X, Y


X, y = load_data()

split_index = len(X)-5
X_train = X[:split_index, :]
X_test = X[split_index:, :]
Y_train = y[:split_index, :]
Y_test = y[split_index:, :]


print("training samples: ", len(X_train))
print("testing samples:",len(X_test))

print("")

regr = MLPRegressor()
regr.fit(X_train, Y_train)


pickle.dump(regr,open("trained_nn","wb"))

print("predicción:")
print(regr.predict(X_test))
print("")
print("objetivos:")
print(Y_test)
print("")
print("score:" , regr.score(X_test, Y_test))


