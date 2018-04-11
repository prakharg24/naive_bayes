import csv
import numpy as np
import random
from sklearn import metrics
import pickle
import sys

def read_csv(filename):
    pixels = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            temp = []
            for i in range(0, 784):
                temp.append((int(row[i]) + 0.0)/255)
            pixels.append(temp)
    return pixels

def getywx(y, w, x, b):
    return y*(np.dot(x, np.transpose(w)) + b)

def get_winn(X, i, j, w, b):
    value = np.dot(X, np.transpose(w)) + b
    if value<0:
        return i
    else:
        return j

def classify(X, all_w, all_b):
    wins = np.zeros([10])
    ind = 0
    for i in range(0, 10):
        for j in range(i+1, 10):
            w = all_w[ind]
            b = all_b[ind]
            winner = get_winn(X, i, j, w, b)
            wins[winner] += 1
            ind += 1    
    max_val = 0
    max_ind = -1
    for i in range(0, 10):
        if(wins[i]>=max_val):
            max_val = wins[i]
            max_ind = i
    return max_ind

te_fea = read_csv(sys.argv[1])

with open('data/weights.pkl', 'rb') as f:
	all_w = pickle.load(f)

with open('data/bias.pkl', 'rb') as f:
	all_b = pickle.load(f)

op_fl = open(sys.argv[2], 'w')

lgt = len(te_fea)
pred_lbs = []
for i in range(0, lgt):
    op_fl.write(str(classify(te_fea[i], all_w, all_b)) + "\n")

op_fl.close()