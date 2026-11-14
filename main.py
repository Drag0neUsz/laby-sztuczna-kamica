import numpy as np
import pandas as pd
from math import exp

tr_set=pd.read_csv("tr_set.csv",sep=",")
test_set=pd.read_csv("test.csv",sep=",")

logs = open("logs.txt", 'w')

# no. wejść = len(kolumny)-1(d)+1(1) 
k = len(tr_set.columns)
# no. ukrytych
h = 6
# no. wyjsciowych [ma kamice 0-1;   nie ma kamicy 0-1]
r = 2

beta = 1
rate = 0.5

v = np.ones(shape=(h+1))
y = np.zeros(shape=(r))
w = np.random.uniform(-0.5, 0.5, size=(h+r, k))


def calc_s(wagi, wejscia):
    return np.dot(wagi, wejscia)

def calc_y(s):
    return 1 / (1 + exp(-beta*s))

# liczymy v
epochs = 1000
for epoch in range(epochs):
    
    errors = []
    for n in range(len(tr_set)):
        delta_w = np.zeros_like(w)
        features = tr_set.values[n][:-1]
        x = np.insert(features, 0, 1)
        target = tr_set.values[n][-1]
        d = np.array([target, 1 - target], dtype=int)


        for j in range(h):
            v[j+1]=calc_y(calc_s(w[j], x))
        # print("v:", v)
        for j in range(r):
            y[j]=calc_y(calc_s(w[h+j], v))
        # print("y:", y)

        #liczymy blad
        # print("d:", d)
        error = 0.5*np.sum((d-y)**2) 
        # print("error:", error)

        # back propagation
        # dla wyjsciowej
        for l in range(r):
            for i in range(h+1): 
                delta_w[h+l][i] = - (d[l] - y[l]) * beta * y[l] * (1- y[l]) * v[i]
        # dla ukrytej
        for i in range(h):
            czlon_sumowy = 0
            for m in range(r):
                czlon_sumowy += - (d[m] - y[m]) * beta * y[m] * (1- y[m]) * w[h+m][i+1]
            for j in range(k):
                delta_w[i][j] = czlon_sumowy * beta * v[i+1] * (1-v[i+1]) * x[j]

        # log
        logs.write(f"epoch = {epoch}: n={n}: error={error} \n")
        errors.append(error)

        # print("x: ", x)
        # print("wagi:", w)
        # print("v: ", v)
        # print("delta_w: ", delta_w)

        w=w-rate*delta_w
    logs.write(f"epoch{epoch} error={np.mean(errors)} \n")
    correct = 0
    tp, fp, fn, tn = 0,0,0,0
    for n in range(len(tr_set)):
        x_test = np.insert(tr_set.values[n][:-1], 0, 1.0)
        target_test = int(tr_set.values[n][-1])
        
        # Forward pass
        v_test = np.ones(shape=(h + 1))
        for j in range(h):
            v_test[j + 1] = calc_y(calc_s(w[j], x_test))
        
        y_test = np.zeros(shape=(r))
        for j in range(r):
            y_test[j] = calc_y(calc_s(w[h + j], v_test))
            
        predicted_class = 1 if y_test[0] > y_test[1] else 0
        if predicted_class == target_test:
            correct += 1
        
        if predicted_class == 1 and target_test == 1:
            tp += 1
        elif predicted_class == 0 and target_test == 0:
            tn += 1
        elif predicted_class == 1 and target_test == 0:
            fp += 1
        elif predicted_class == 0 and target_test == 1:
            fn += 1
    accuracy = (correct / len(tr_set)) * 100
    logs.write(f"====accuracy na tr_set: {accuracy:.2f}% ({correct}/{len(tr_set)})\n")
    logs.write(f"tp={tp}, fp={fp}, fn={fn}, tn={tn}")
    
    correct = 0
    tp, fp, fn, tn = 0,0,0,0
    for n in range(len(test_set)):
        x_test = np.insert(test_set.values[n][:-1], 0, 1.0)
        target_test = int(test_set.values[n][-1])
        
        # Forward pass
        v_test = np.ones(shape=(h + 1))
        for j in range(h):
            v_test[j + 1] = calc_y(calc_s(w[j], x_test))
        
        y_test = np.zeros(shape=(r))
        for j in range(r):
            y_test[j] = calc_y(calc_s(w[h + j], v_test))
            
        predicted_class = 1 if y_test[0] > y_test[1] else 0
        if predicted_class == target_test:
            correct += 1
        
        if predicted_class == 1 and target_test == 1:
            tp += 1
        elif predicted_class == 0 and target_test == 0:
            tn += 1
        elif predicted_class == 1 and target_test == 0:
            fp += 1
        elif predicted_class == 0 and target_test == 1:
            fn += 1
    accuracy = (correct / len(test_set)) * 100
    logs.write(f"\naccuracy na test_set: {accuracy:.2f}% ({correct}/{len(test_set)})\n")
    logs.write(f"tp={tp}, fp={fp}, fn={fn}, tn={tn}\n===============================================\n")

correct = 0
tp, fp, fn, tn = 0,0,0,0
for n in range(len(test_set)):
    x_test = np.insert(test_set.values[n][:-1], 0, 1.0)
    target_test = int(test_set.values[n][-1])
    
    # Forward pass
    v_test = np.ones(shape=(h + 1))
    for j in range(h):
        v_test[j + 1] = calc_y(calc_s(w[j], x_test))
    
    y_test = np.zeros(shape=(r))
    for j in range(r):
        y_test[j] = calc_y(calc_s(w[h + j], v_test))
        
    predicted_class = 1 if y_test[0] > y_test[1] else 0
    if predicted_class == target_test:
        correct += 1
    
    if predicted_class == 1 and target_test == 1:
        tp += 1
    elif predicted_class == 0 and target_test == 0:
        tn += 1
    elif predicted_class == 1 and target_test == 0:
        fp += 1
    elif predicted_class == 0 and target_test == 1:
        fn += 1
    logs.write(f"test_set n={n}: target={target_test}: predicted={predicted_class}\n")
accuracy = (correct / len(test_set)) * 100
logs.write(f"\naccuracy na test_set: {accuracy:.2f}% ({correct}/{len(test_set)})\n")
logs.write(f"tp={tp}, fp={fp}, fn={fn}, tn={tn}\n")



logs.write(f"koncowe parametry: \n {w}")

logs.close()