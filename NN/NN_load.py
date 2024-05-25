from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def load_data():
    num_cols = 56 * 4 + 3
    data = np.genfromtxt("../generacion_datos/datos.txt")
    X = data[:, :num_cols - 3]
    Y = data[:, -3:]
    return X, Y


def load_predictor():
    with open("trained_nn","rb") as f:
        model = pickle.load(f)
    return model

X, y = load_data()
model = load_predictor()

print("predicted:")
out1,out2,out3 = model.predict(X[:1,:])[0];
print(out1,out2,out3)
print("")
print("objetivos:")
print(y[:1,:])




