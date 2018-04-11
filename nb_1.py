import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys
import pickle

def read_document(filename):
	file = open(filename, 'r', errors='ignore')
	rev = file.readlines()
	return rev

def rat_to_label(rating):
    if rating<5:
        return rating - 1
    else:
        return rating - 3

def label_to_rat(label):
    if label<4:
        return label + 1
    else:
        return label + 3


test_rev_un = read_document(sys.argv[1])

print("Loading vocabulary")

with open('data/vocab_dict_1.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

cleaned_test_rev = []
print("Vectorizing Test data")
for sent in test_rev_un:
    cleaned_test_rev.append(sent.split())

print("Loading Naive Bayes Matrices")

with open('data/log_matrix_1.pkl', 'rb') as f:
    log_matrix = pickle.load(f)

with open('data/log_class_1.pkl', 'rb') as f:
    log_class = pickle.load(f)

print("Predicting data")

output_file = open(sys.argv[2], 'w')

for j in range(len(cleaned_test_rev)):
    sent = cleaned_test_rev[j]
    max_class = 0
    max_prob = 0.0
    for i in range(0, 8):
        sum_prob = log_class[i]
        for it in range(len(sent)):
            if sent[it] in vocab_dict:
                sum_prob = sum_prob + log_matrix[vocab_dict[sent[it]]][i]
        if i==0:
            max_prob = sum_prob
            max_class = i
        if sum_prob>max_prob:
            max_prob = sum_prob
            max_class = i
    output_file.write(str(label_to_rat(max_class))  + "\n")

output_file.close()