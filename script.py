import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions import hypothesis, koszt, gradient_descent
data = pd.read_csv('insurance.csv')
with open('params.json', 'r') as f:
    params = json.load(f)
alpha = params['alpha']
num_iters = params['num_iters']
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
X = data[['age', 'sex', 'bmi', 'children', 'smoker']].copy()
y = data['charges'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mean_values = X_train.mean()
std_values = X_train.std()
X_train = (X_train - mean_values) / std_values
X_test = (X_test - mean_values) / std_values
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
theta = np.zeros((X_train.shape[1], 1))
print(X_train)
theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)
y_pred = np.dot(X_test, theta)
plt.scatter(y_test, y_pred)
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color='black')
plt.xlabel('Wartosci rzeczywiste')
plt.ylabel('Wartosci przewidywane')
plt.title('Rzeczywiste vs przewidywane')
plt.show()
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / ss_tot
print("Theta:")
print(theta)
print("\nR2:")
print(r2)