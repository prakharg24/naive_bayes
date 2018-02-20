import numpy as np
import random

def transform(matr, class_prob):
	matr = matr + 1
	denom = np.sum(matr, axis = 0)
	denom = np.reshape(denom, [1,8])
	print(matr)
	print(denom)
	print(class_prob)
	log_den = np.log(denom)
	log_prob = np.log(class_prob)
	log_matr = np.log(matr)
	log_matr = log_matr - log_den
	return log_matr, log_prob

def naive_bayes(matr, x, y):
	return matr[x][y]

def is_punc(char):
	puncs = ['.', ',', '?', ':', ';', '\'', '\"', '(', ')', '!', '/', '\\', '-', '*', '$']
	for ch in puncs:
		if char==ch:
			return True
	if char<='9' and char>='0':
		return True
	return False

def clean_data(string):
	ans_str = ""
	i = 0
	while i < len(string):
		if string[i]=='<':
			while i < len(string) and string[i]!='>':
				i = i + 1
			ans_str = ans_str + " "
			i = i+1
			continue
		if is_punc(string[i]):
			ans_str = ans_str + " "
			i = i+1
			continue
		ans_str = ans_str + string[i]
		i = i + 1
	return ans_str.split()

def get_lower(str_list):
	ans = []
	for st in str_list:
		ans.append(st.lower())
	return ans

def read_document(filename):
	file = open(filename, 'r', errors='ignore')
	rev = file.readlines()
	return rev

def create_vocab(lst):
	global vocab
	for word in lst:
		vocab.add(word)

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


train_rev = read_document("imdb_train_text_clean.txt")
cleaned_rev = []
vocab = set()
print("Cleaning data")
for sent in train_rev:
	# temp_sen = clean_data(sent)
	# lower_sen = get_lower(temp_sen)
	cleaned_rev.append(sent.split())
	create_vocab(sent.split())

print("Creating vocab")

vocab_dict = {}
ite = 0
for ele in vocab:
	vocab_dict[ele] = ite
	ite = ite + 1

new_matr = np.zeros([ite,8])

train_labels_str = read_document("imdb/imdb_train_labels.txt")

train_labels = []

for i in range(len(train_labels_str)):
	train_labels.append(int(train_labels_str[i]))

class_prob = np.zeros([8])

for i in range(len(train_labels)):
	class_prob[rat_to_label(train_labels[i])] += 1

max_val = 0
max_cls = 0
for i in range(len(class_prob)):
	if class_prob[i]>max_val:
		max_val = class_prob[i]
		max_cls = i

class_prob = class_prob/len(train_labels)

print("Creating Matrix")

for i in range(len(train_labels)):
	label = rat_to_label(train_labels[i])
	rev = cleaned_rev[i]
	for word in rev:
		row = vocab_dict.get(word)
		new_matr[row][label] += 1

print("Log Matrix creation")

log_matrix , log_class = transform(new_matr, class_prob)


test_rev = read_document("imdb_test_text_clean.txt")
cleaned_test_rev = []
print("Cleaning Test data")
for sent in test_rev:
	# temp_sen = clean_data(sent)
	# lower_sen = get_lower(temp_sen)
	cleaned_test_rev.append(sent.split())

test_labels_str = read_document("imdb/imdb_test_labels.txt")

test_labels = []

print("Matching data")

for i in range(len(test_labels_str)):
	test_labels.append(int(test_labels_str[i]))

total_count = 0
corr_count_nb = 0
corr_count_rn = 0
corr_count_mj = 0

confusion = np.zeros([8, 8])

for j in range(len(cleaned_test_rev)):
	if j%100==0:
		print(j)
	sent = cleaned_test_rev[j]
	max_class = 0
	max_prob = 0.0
	for i in range(0, 8):
		sum_prob = log_class[i]
		for word in sent:
			if word in vocab_dict.keys():
				sum_prob = sum_prob + log_matrix[vocab_dict[word]][i]
		if i==0:
			max_prob = sum_prob
			max_class = i
		if sum_prob>max_prob:
			max_prob = sum_prob
			max_class = i
	if label_to_rat(max_cls)==test_labels[j]:
		corr_count_mj = corr_count_mj + 1
	rand_class = random.randint(0, 8)
	if label_to_rat(rand_class)==test_labels[j]:
		corr_count_rn = corr_count_rn + 1
	if label_to_rat(max_class)==test_labels[j]:
		corr_count_nb = corr_count_nb + 1
	total_count = total_count + 1
	confusion[max_class][rat_to_label(test_labels[j])] += 1

for i in range(0, 8):
	for j in range(0, 8):
		print(int(confusion[i][j]), "\t", end='')
	print()

print("Accuracy NB :", (corr_count_nb+0.0)/total_count)
print("Accuracy RN :", (corr_count_rn+0.0)/total_count)
print("Accuracy MJ :", (corr_count_mj+0.0)/total_count)
