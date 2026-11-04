from random import random
from math import exp
import numpy as np
import matplotlib.pyplot as plt

tr_set = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))

w_l1 = np.random.uniform(-1, 1, size=(2, 3))

w_l2 = np.random.uniform(-1, 1, size=3)

errors = []

beta = 1
rate = 0.5

# summation term
def calc_s(w, x):
    return np.dot(w, x)

# fcja aktywacji
def sigmoid(s):
    global beta
    return 1 / (1 + exp(-beta*s))

def test_all_correct(w_l1, w_l2):
    global tr_set
    accuracy = 0
    results = []
    for k in range(len(tr_set)):

        
        x = np.array([1]+list(tr_set[k][:-1]))
        d = tr_set[k][-1]
        v1 = sigmoid(calc_s(w_l1[0], x))
        v2 = sigmoid(calc_s(w_l1[1], x))
        y = sigmoid(calc_s(w_l2, np.array([1, v1, v2])))
                        #y, class, d
        results.append([y, int(y>0.5), d])
        accuracy += int(y>0.5) == d

    return np.array(results).reshape(len(tr_set), 3), accuracy == len(tr_set)

print("STARTING WEIGHTS: ")
print("w_l1: ", w_l1)
print("w_l2: ", w_l2)

correct = False
num_iter = 10000
n = 0
while not correct and n < num_iter:
    delta_l1 = np.zeros_like(w_l1)
    delta_l2 = np.zeros_like(w_l2)
    n += 1
    for k in range(len(tr_set)):
        x = np.array([1]+list(tr_set[k][:-1]))
        d = tr_set[k][-1]

        # print("tr set: ", tr_set[k])
        s_l1=np.array([0]*len(w_l1), dtype=float)
        v_l1=np.array([0]*len(w_l1), dtype=float)
        s_l2=0.
        
        # forward pass
        for i in range(len(w_l1)):
            w = w_l1[i]
            # print(f"w_l1[{i}]: {w}, x: {x}")
            s_l1[i] = calc_s(w, x)
            # print(f"s_l1[{i}]: {s_l1[i]}")
            v_l1[i] = sigmoid(s_l1[i])
            # print(f"v_l1[{i}]: {v_l1[i]}", "\n")


        # forward pass v2 do wyjscia
        v = np.array([1]+list(v_l1))

        # print(f"w_l2: {w_l2}, v: {v}")
        s_l2 = calc_s(w_l2, v)
        # print(f"s_l2: {s_l2}")
        y = sigmoid(s_l2)
        # print(f"y: {y}", "\n")

        error = d - y
        mse = 0.5 * error**2
        errors.append(mse)
        # print(f"mse: {mse}", "\n")

        # delta z l2
        d_out = -error * (beta * y) * (1-y)
        delta_l2 += d_out * v
        # print(f"delta_l2: {delta_l2}", "\n")

        for i in range(len(w_l1)):
            delta_l1[i] += d_out * w_l2[i+1] * beta * v_l1[i] * (1-v_l1[i]) * x

        # print(f"delta_l1: {delta_l1}", "\n")
        # print("--------------------------------")

    w_l1 = w_l1 - rate * delta_l1/4
    w_l2 = w_l2 - rate * delta_l2/4
    _, correct = test_all_correct(w_l1, w_l2)
print("CORRECT: ", correct)
print("NUMBER OF ITERATIONS: ", n)
print("MSE: ", mse)
print("WEIGHTS: ")
print("w_l1: ", w_l1)
print("w_l2: ", w_l2)

