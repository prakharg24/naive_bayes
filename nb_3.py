import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys
import pickle

#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

# function that takes an input file and performs stemming to generate the output file
def getStemmedDocument(doc):
    raw = doc.replace("<br /><br />", " ")
    tokens = tokenizer.tokenize(raw)
    temp = []
    for i in range(len(tokens)-2):
        if tokens[i]=='not':
            tokens[i+1] = 'not_' + tokens[i+1]
            tokens[i+2] = 'not_' + tokens[i+2]
    for word in tokens:
        if any(x.isupper() for x in word):
            temp.append(word)
    for i in range(len(tokens)-1):
        temp.append(tokens[i]+"_"+tokens[i+1])
    for ele in temp:
        tokens.append(ele)
    raw = []
    for ele in tokens:
        raw.append(ele.lower())
    stopped_tokens = [token for token in raw if token not in en_stop]
    stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
    documentWords = ' '.join(stemmed_tokens)
    return documentWords

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

print("Pre-processing the test data")

test_rev = []
for sent in test_rev_un:
    test_rev.append(getStemmedDocument(sent))

print("Loading vocabulary")

with open('data/vocab_dict_3.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

cleaned_test_rev = []
print("Vectorizing Test data")
for sent in test_rev:
    cleaned_test_rev.append(sent.split())

print("Loading Naive Bayes Matrices")

with open('data/log_matrix_3.pkl', 'rb') as f:
    log_matrix = pickle.load(f)

with open('data/log_class_3.pkl', 'rb') as f:
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