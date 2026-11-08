import numpy as np
import pandas as pd
from math import exp

tr_set=pd.read_csv("tr_set.csv",sep=",")
tr_set = (tr_set - tr_set.min()) / (tr_set.max() - tr_set.min())
test_set=pd.read_csv("test.csv",sep=",")

print(tr_set.head(1))
# no. wejść = len(kolumny)-1(target)+1(1) 
k = len(tr_set.columns)
# no. ukrytych
h = 6
# no. wyjsciowych
r = 2

beta = 1
rate = 0.5

v = np.ones(shape=(k, 1))
y = np.zeros(shape=(r, 1))
w = np.random.uniform(-0.5, 0.5, size=(h+r, k))

def calc_s(wagi, wejscia):
    return np.dot(wagi, wejscia)

def calc_y(s):
    return 1 / (1 + exp(-beta*s))

# liczymy v
i = 0

x = tr_set.values[i]
x[-1]=1


for k in range(h):
    v[k]=calc_s(w[k], x)
print(v)
print()
for k in range(r):
    y[k]=calc_s(w[h+k], v)
print(y)
