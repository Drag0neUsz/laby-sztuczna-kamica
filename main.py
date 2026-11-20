import time
import numpy as np
import pandas as pd
from math import exp
from plot_utils import plot_errors, format_ascii_cm

start_time = time.perf_counter()

tr_set=pd.read_csv("tr_set.csv",sep=",")
test_set=pd.read_csv("test.csv",sep=",")

logs = open("logs.txt", 'w')
logs2 = open("logs2.txt", 'w')
temp_log = open("temp.txt", 'w')

# no. wejść = len(kolumny)-1(d)+1(1) 
k = len(tr_set.columns)
# no. ukrytych
h = 6
# no. wyjsciowych [ma kamice 0-1;   nie ma kamicy 0-1]
r = 2

beta = 1
rate = 0.5

np.random.seed(21)

v = np.ones(shape=(h+1))
y = np.zeros(shape=(r))
w_h = np.random.uniform(-0.5, 0.5, size=(h, k))
w_y = np.random.uniform(-0.5, 0.5, size=(r, h+1))


def calc_s(wagi, wejscia):
    return np.dot(wagi, wejscia)

def calc_y(s):
    return 1 / (1 + exp(-beta*s))

# liczymy v
epochs = 1000
epochs_stat, tr_errors, te_errors = [], [], []
for epoch in range(epochs):
    
    errors = []
    for n in range(len(tr_set)):
        delta_w_h = np.zeros_like(w_h)
        delta_w_y = np.zeros_like(w_y)
        features = tr_set.values[n][:-1]
        x = np.insert(features, 0, 1)
        target = tr_set.values[n][-1]
        d = np.array([target, 1 - target], dtype=int)


        for j in range(h):
            v[j+1]=calc_y(calc_s(w_h[j], x))
        # print("v:", v)
        for j in range(r):
            y[j]=calc_y(calc_s(w_y[j], v))
        # print("y:", y)

        #liczymy blad
        # print("d:", d)
        error = 0.5*np.sum((d-y)**2) 
        # print("error:", error)

        # back propagation
        # dla wyjsciowej
        for l in range(r):
            for i in range(h+1): 
                delta_w_y[l][i] = - (d[l] - y[l]) * beta * y[l] * (1- y[l]) * v[i]
        # dla ukrytej
        for i in range(h):
            czlon_sumowy = 0
            for m in range(r):
                czlon_sumowy += - (d[m] - y[m]) * beta * y[m] * (1- y[m]) * w_y[m][i+1]
            for j in range(k):
                delta_w_h[i][j] = czlon_sumowy * beta * v[i+1] * (1-v[i+1]) * x[j]

        # log
        logs.write(f"epoch = {epoch}: n={n}: error={error} \n")
        errors.append(error)

        # print("x: ", x)
        # print("wagi:", w)
        # print("v: ", v)
        # print("delta_w: ", delta_w)

        w_h=w_h-rate*delta_w_h
        w_y=w_y-rate*delta_w_y
    logs.write(f"epoch{epoch} error={np.mean(errors)} \n")
    correct = 0
    tp, fp, fn, tn = 0,0,0,0
    for n in range(len(tr_set)):
        x_test = np.insert(tr_set.values[n][:-1], 0, 1.0)
        target_test = int(tr_set.values[n][-1])
        
        # Forward pass
        v_test = np.ones(shape=(h + 1))
        for j in range(h):
            v_test[j + 1] = calc_y(calc_s(w_h[j], x_test))
        
        y_test = np.zeros(shape=(r))
        for j in range(r):
            y_test[j] = calc_y(calc_s(w_y[j], v_test))
            
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
    tr_accuracy = accuracy
    logs.write(f"====accuracy na tr_set: {accuracy:.2f}% ({correct}/{len(tr_set)})\n")
    logs.write(f"tp={tp}, fp={fp}, fn={fn}, tn={tn}")
    tr_cm = (tp, fp, fn, tn)
    
    correct = 0
    tp, fp, fn, tn = 0,0,0,0
    test_errors = []
    for n in range(len(test_set)):
        x_test = np.insert(test_set.values[n][:-1], 0, 1.0)
        target_test = int(test_set.values[n][-1])
        d_test = np.array([target_test, 1 - target_test], dtype=int)
        
        # Forward pass
        v_test = np.ones(shape=(h + 1))
        for j in range(h):
            v_test[j + 1] = calc_y(calc_s(w_h[j], x_test))
        
        y_test = np.zeros(shape=(r))
        for j in range(r):
            y_test[j] = calc_y(calc_s(w_y[j], v_test))
            
        test_errors.append(0.5*np.sum((d_test-y_test)**2))
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

    if epoch % 25 == 0 or epoch == epochs - 1:
        epochs_stat.append(epoch)
        tr_errors.append(np.mean(errors))
        te_errors.append(np.mean(test_errors))
        logs2.write(format_ascii_cm(epoch, tr_cm, (tp, fp, fn, tn)))

    if epoch in (0, 500, epochs - 1):
        temp_log.write(format_ascii_cm(epoch, tr_cm, (tp, fp, fn, tn)))
        temp_log.write(f"accuracy na tr_set: {tr_accuracy:.2f}%\n")
        temp_log.write(f"accuracy na test_set: {accuracy:.2f}%\n")

correct = 0
tp, fp, fn, tn = 0,0,0,0
for n in range(len(test_set)):
    x_test = np.insert(test_set.values[n][:-1], 0, 1.0)
    target_test = int(test_set.values[n][-1])
    
    # Forward pass
    v_test = np.ones(shape=(h + 1))
    for j in range(h):
        v_test[j + 1] = calc_y(calc_s(w_h[j], x_test))
    
    y_test = np.zeros(shape=(r))
    for j in range(r):
        y_test[j] = calc_y(calc_s(w_y[j], v_test))
        
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



logs.write(f"koncowe parametry: \n {w_h} \n {w_y}")

logs.close()
logs2.close()
temp_log.close()

plot_errors(epochs_stat, tr_errors, te_errors, h)
print(f"Czas wykonania: {time.perf_counter() - start_time:.2f} s")