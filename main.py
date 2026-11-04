import numpy as np
import pandas as pd
from math import exp

data=pd.read_csv("data.csv",sep=",")

print(data.head(5))
# no. wejść = len(kolumny)-1+1 ale 1 z kolumn to target więc jej nie liczymy ale i tak trzeba dodac 1
k = len(data.columns)
# no. ukrytych
h = 6
# no. wyjsciowych
r = 2

beta = 1
rate = 0.5

w = np.random.uniform(-1, 1, size=(h+r, k))

def calc_s(wagi, wejscia):
    return np.dot(wagi, wejscia)

def calc_y(s):
    return 1 / (1 + exp(-beta*s))

