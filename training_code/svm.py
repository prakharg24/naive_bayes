
# coding: utf-8

# In[130]:


import csv
import numpy as np
import random
from sklearn import metrics
import pickle


# In[103]:


def read_csv(filename):
    pixels = []
    labels = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            temp = []
            for i in range(0, 784):
                temp.append((int(row[i]) + 0.0)/255)
            pixels.append(temp)
            labels.append([int(row[784])])
    return pixels, labels


# In[104]:


tr_fea, tr_lbs = read_csv('train.csv')
te_fea, te_lbs = read_csv('test.csv')


# In[105]:


def getywx(y, w, x, b):
    return y*(np.dot(x, np.transpose(w)) + b)

# X -> m*n
# y -> m*1
# w -> 1*n
def svm_classifier(X, y, f_len, it, bt):
    w = np.zeros([1, f_len])
    b = 0
    m = len(X)
    for i in range(0, it):
        x_ran = []
        y_ran = []
        prb = random.sample(range(0, m), bt)
        for j in range(0, bt):
            if getywx(y[prb[j]], w, X[prb[j]], b)<1:
                x_ran.append(X[prb[j]])
                y_ran.append(y[prb[j]])
        eta = 1.0/(i+1)
        w = (1-eta)*w + (eta/bt)*np.dot(np.transpose(y_ran), x_ran)
        b = b + (eta/bt)*np.sum(y_ran)
    return w, b

def create_train(X, y, i, j):
    X_out = []
    y_out = []
    m = len(y)
    for it in range(0, m):
        if(y[it][0]==i):
            X_out.append(X[it])
            y_out.append([-1])
        elif(y[it][0]==j):
            X_out.append(X[it])
            y_out.append([1])
    return X_out, y_out        
    
def all_classifiers(X, y, f_len, it, bt):
    all_w = []
    all_b = []
    for i in range(0, 10):
        for j in range(i+1, 10):
            print(i, " ", j)
            X_tp, y_tp = create_train(X, y, i, j)
            w_tp, b_tp = svm_classifier(X_tp, y_tp, f_len, it, bt)
            all_w.append(w_tp)
            all_b.append(b_tp)
    return all_w, all_b


# In[106]:


def get_winn(X, i, j, w, b):
    value = np.dot(X, np.transpose(w)) + b
    if value<0:
        return i
    else:
        return j


# In[107]:


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
#     print(wins)
    
    max_val = 0
    max_ind = -1
    for i in range(0, 10):
        if(wins[i]>=max_val):
            max_val = wins[i]
            max_ind = i
    return max_ind


# In[123]:


all_w, all_b = all_classifiers(tr_fea, tr_lbs, 784, 10000, 100)


# In[124]:


lgt = len(te_fea)
print(lgt)
pred_lbs = []
for i in range(0, lgt):
    if (i%1000==0):
        print(i)
    pred_lbs.append(classify(te_fea[i], all_w, all_b))

print(metrics.confusion_matrix(te_lbs, pred_lbs))
print(metrics.accuracy_score(te_lbs, pred_lbs))


# In[125]:


lgt = len(tr_fea)
print(lgt)
pred_lbs = []
for i in range(0, lgt):
    if (i%1000==0):
        print(i)
    pred_lbs.append(classify(tr_fea[i], all_w, all_b))
    
print(metrics.confusion_matrix(tr_lbs, pred_lbs))
print(metrics.accuracy_score(tr_lbs, pred_lbs))


# In[131]:


with open('weights.pkl', 'wb') as f:
    pickle.dump(all_w, f)
with open('bias.pkl', 'wb') as f:
    pickle.dump(all_b, f)

