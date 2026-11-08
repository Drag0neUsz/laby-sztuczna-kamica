import numpy as np
import pandas as pd
from math import exp

tr_set=pd.read_csv("tr_set.csv",sep=",")
tr_set = (tr_set - tr_set.min()) / (tr_set.max() - tr_set.min())
test_set=pd.read_csv("test.csv",sep=",")

print(tr_set.head(1))
# no. wejść = len(kolumny)-1(d)+1(1) 
k = len(tr_set.columns)
# no. ukrytych
h = 6
# no. wyjsciowych [ma kamice 0-1;   nie ma kamicy 0-1]
r = 2

beta = 1
rate = 0.5

v = np.ones(shape=(k))
y = np.zeros(shape=(r))
w = np.random.uniform(-0.5, 0.5, size=(h+r, k))
delta_w = np.zeros_like(w)

def calc_s(wagi, wejscia):
    return np.dot(wagi, wejscia)

def calc_y(s):
    return 1 / (1 + exp(-beta*s))

# liczymy v
n = 0

features = tr_set.values[n][:-1]
x = np.insert(features, 0, 1)
target = tr_set.values[n][-1]
d = np.array([target, 1 - target], dtype=int)


for j in range(1, h+1):
    v[j]=calc_y(calc_s(w[j], x))
print("v:", v)
print()
for j in range(r):
    y[j]=calc_y(calc_s(w[h+j], v))
print("y:", y)

#liczymy blad
print("d:", d)
error = 0.5*np.sum((d-y)**2) 
print("error:", error)

# back propagation
# dla wyjsciowej
for l in range(r):
    for i in range(k): 
        delta_w[h+l][i] = - (d[l] - y[l]) * beta * y[l] * (1- y[l]) * v[i]
# dla ukrytej
for i in range(h):
    czlon_sumowy = 0
    for m in range(r):
        czlon_sumowy += - (d[m] - y[m]) * beta * y[m] * (1- y[m]) * w[h+m][i]
    for j in range(k):
        delta_w[i][j] = czlon_sumowy * beta * v[i+1] * (1-v[i+1]) * x[j]

print("x: ", x)
print("wagi:", w)
print("v: ", v)
print("delta_w: ", delta_w)
