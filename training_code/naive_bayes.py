
# coding: utf-8

# In[19]:


import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys

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


# In[20]:


train_rev_un = read_document("imdb/imdb_train_text.txt")
train_labels_str = read_document("imdb/imdb_train_labels.txt")
test_rev_un = read_document("imdb/imdb_test_text.txt")
test_labels_str = read_document("imdb/imdb_test_labels.txt")


# In[21]:


def transform(matr, class_prob):
	global vocab_dict
	lgt = len(matr)
	i=0
	while(i<lgt):
		flag = 0
		if(i%10000==0):
			print(i, " ", lgt)
		for j in range(0, 8):
			if(matr[i][j]>5):
				flag = 1
		if (flag==0):
			for j in range(0, 8):
				matr[i][j]=0
		i += 1
	matr = matr + 1
	denom = np.sum(matr, axis = 0)
	denom = np.reshape(denom, [1,8])
	log_den = np.log(denom)
	log_prob = np.log(class_prob)
	log_matr = np.log(matr)
	log_matr = log_matr - log_den
	return log_matr, log_prob

def naive_bayes(matr, x, y):
	return matr[x][y]

def create_vocab(lst):
    global vocab
    lng = len(lst)
    temp = set()
    for i in range(0, lng):
        vocab.add(lst[i])

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


# In[22]:


train_labels = []

for i in range(len(train_labels_str)):
	train_labels.append(int(train_labels_str[i]))

class_prob = np.zeros([8])

for i in range(len(train_labels)):
	class_prob[rat_to_label(train_labels[i])] += 1
    
test_labels = []

for i in range(len(test_labels_str)):
	test_labels.append(int(test_labels_str[i]))

train_rev = []
for sent in train_rev_un:
    train_rev.append(getStemmedDocument(sent))

test_rev = []
for sent in test_rev_un:
    test_rev.append(getStemmedDocument(sent))


# In[23]:


cleaned_rev = []
vocab = set()
print("Cleaning data")
for sent in train_rev:   
	cleaned_rev.append(sent.split())
	create_vocab(sent.split())

print("Creating vocab")

vocab_dict = {}
ite = 0
for ele in vocab:
	vocab_dict[ele] = ite
	ite = ite + 1

cleaned_test_rev = []
print("Cleaning Test data")
for sent in test_rev:
	cleaned_test_rev.append(sent.split())


# In[24]:


max_val = -1
max_cls = 0
for i in range(len(class_prob)):
	if class_prob[i]>max_val:
		max_val = class_prob[i]
		max_cls = i

class_prob = class_prob/len(train_labels)

print("Creating Matrix")

new_matr = np.zeros([ite,8])

print(np.shape(cleaned_rev))

for i in range(len(train_labels)):
	label = rat_to_label(train_labels[i])
	rev = cleaned_rev[i]
	lngt = len(rev)
	for j in range(lngt):
		row = vocab_dict.get(rev[j])
		new_matr[row][label] += 1

print("Log Matrix creation")

log_matrix , log_class = transform(new_matr, class_prob)

ite = 0
for ele in vocab_dict.keys():
	vocab_dict[ele] = ite
	ite = ite + 1


# In[25]:


total_count = 0
corr_count_nb = 0

confusion = np.zeros([8, 8])

for j in range(len(cleaned_test_rev)):
	if j%5000==0:
		print(j)
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
	if label_to_rat(max_class)==test_labels[j]:
		corr_count_nb = corr_count_nb + 1
	total_count = total_count + 1
	confusion[max_class][rat_to_label(test_labels[j])] += 1

for i in range(0, 8):
	for j in range(0, 8):
		print(int(confusion[i][j]), "\t", end='')
	print()

print("Accuracy NB :", (corr_count_nb+0.0)/total_count)


# In[26]:


import pickle
with open('log_matrix_2.pkl', 'wb') as f:
    pickle.dump(log_matrix, f)
with open('log_class_2.pkl', 'wb') as f:
    pickle.dump(log_class, f)


# In[27]:


with open('vocab_dict_2.pkl', 'wb') as f:
    pickle.dump(vocab_dict, f)

