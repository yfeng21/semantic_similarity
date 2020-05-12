import nltk
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
import itertools
import numpy as np 
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine 
import spacy
import main 
import csv 

filename = 'cleaned_testset.csv'
data = pd.read_csv(filename)
mr = data['mr']
sents = data['ref']
nlp = spacy.load("en_core_web_md")
MR2SENT = main.group_sents_by_mr(mr, sents)


def tokenize(sent):
	# input a sent and output a list
	tokens = nlp(sent)
	tokenized_words = [token.text for token in tokens]
	# tokenizer = nltk.RegexpTokenizer(r"\w+")
    # tokenized_words = tokenizer.tokenize(sent)
	return tokenized_words

tokenized_sents = [tokenize(sent) for sent in sents]

all_tokens = itertools.chain.from_iterable(tokenized_sents)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
id_to_word = {idx: token for idx, token in word_to_id.items()}
print(len(word_to_id))
print(len(id_to_word))

#token_ids = [[word_to_id[token] for token in token_sent] for token_sent in tokenized_sents]

def get_grouped_sent():
	grouped_vecs = {}

	for i in range(len(mr)):
		if mr[i] not in grouped_vecs:
			sent_vector = get_one_hot(sents[i])
			grouped_vecs[mr[i]] = [sent_vector]
		else:
			sent_vector = get_one_hot(sents[i])
			grouped_vecs[mr[i]].append(sent_vector)

	return grouped_vecs

	#print(grouped_sent)
def get_one_hot(sent):
	tokenized_sent = tokenize(sent) 
	token_id = [word_to_id[token] for token in tokenized_sent]
	sent_vec = [0 for i in range(len(word_to_id))]
	for index in token_id:
		sent_vec[index] = 1
	return sent_vec

def compute_inter_group_sim(mr1, mr2, mr2sent):
	sent_vec_group1 = mr2sent[mr1]
	sent_vec_group2 = mr2sent[mr2]
	all_sim = 0
	count = 0
	sent_group1 = MR2SENT[mr1]
	sent_group2 = MR2SENT[mr2]


	for i in range(len(sent_vec_group1)):
		for j in range(len(sent_vec_group2)):
			vec1 = sent_vec_group1[i]
			vec2 = sent_vec_group2[j]

			print(sent_group1[i]+'\t'+sent_group2[j]+'\t'+str(1-cosine(vec1,vec2)))

			all_sim += cosine(vec1, vec2)
			count += 1
	avg_sim = all_sim / count 
	avg_sim = 1 - avg_sim
	print(mr1 + " *** " + mr2 + " inter group similarity is "+str(avg_sim))


grouped_vecs = get_grouped_sent()
#print(grouped_vecs)
k1 = "name[Alimentum], area[city centre], familyFriendly[no]"
k1 = 'name[Alimentum], area[riverside], familyFriendly[no]'



# sent1 = grouped_vecs[k1][2]
# sent2 = grouped_vecs[k1][3]
#similarity = cosine(sent1, sent2)
#print(similarity)

# sent_a = get_one_hot(sent1)
# sent_b = get_one_hot(sent2)
# print(cosine(sent1,sent2))


def compute_avg_sim(key):
	# given the key, compute average similarity for all sentneces
	sent_vecs = grouped_vecs[key]
	sents = MR2SENT[key]
	all_sim = 0
	count = 0

	for i in range(len(sent_vecs)-1):
		for j in range(i+1, len(sent_vecs)):
			sent_vec1 = sent_vecs[i]
			sent_vec2 = sent_vecs[j]
			sim = cosine(sent_vec1, sent_vec2)

			print(sents[i]+'\t'+sents[j]+'\t'+str(1-sim))

			all_sim += sim 
			count += 1
	
	avg_sim = all_sim / count 
	
	return 1 - avg_sim 

def compute_avg_sim_var_onehot(key):
	# given the key, compute average similarity for all sentneces
	sent_vecs = grouped_vecs[key]
	sents = MR2SENT[key]
	all_sim = []

	for i in range(len(sent_vecs)-1):
		for j in range(i+1, len(sent_vecs)):
			sent_vec1 = sent_vecs[i]
			sent_vec2 = sent_vecs[j]
			sim = cosine(sent_vec1, sent_vec2)
			cosine_sim = 1 - sim
			all_sim.append(cosine_sim)
	
	avg_sim = sum(all_sim) / len(all_sim)
	var = np.var(all_sim)

	return avg_sim, var

negation_file = "negation_sents.csv"
neg_data = pd.read_csv(negation_file)
mr1 = neg_data['mr1']
nl1 = neg_data['nl1']
mr2 = neg_data['mr2']
nl2 = neg_data['nl2']
output_name = "negation_results/onehot_negation2.csv"
output_csv = open(output_name, 'w')
csvwriter = csv.writer(output_csv)
csvwriter.writerow(['mr1', 'nl1', 'mr2', 'nl2', 'one-hot score'])
for i in range(len(nl1)):
	if i < 84:
		continue
	sent1 = nl1[i]
	sent2 = nl2[i]
	vec1 = get_one_hot(sent1)
	vec2 = get_one_hot(sent2)
	sim = 1 - cosine(vec1, vec2)
	csvwriter.writerow([mr1[i], nl1[i], mr2[i], nl2[i], sim])

# csvwriter.writerow(['mr', 'one-hot score', 'one-hot variance', '# of sentences'])
# for mr, sents in MR2SENT.items():
# 	if len(sents) <= 1:
# 		continue
# 	avg_sim, var = compute_avg_sim_var_onehot(mr)
# 	# print(avg_sim, var)
# 	csvwriter.writerow([mr, avg_sim, var, len(sents)])


# get in group similarity
# for key in grouped_vecs.keys():
# 	# key = mr[i]
# 	avg_sim = compute_avg_sim(key)
# 	print(key + ' : %s' % float('%.2g' % avg_sim) ) 


# mr1 = "name[Alimentum], area[city centre], familyFriendly[yes]"
# mr2 = 'name[Alimentum], area[city centre], familyFriendly[no]'
# mr3 = "name[Aromi], eatType[coffee shop], food[Chinese], customer rating[average], area[city centre], familyFriendly[yes]"
# print(len(grouped_vecs[mr1]))

# print("######### MR1, MR2############")
# compute_inter_group_sim(mr1, mr2, grouped_vecs)

# print("######### MR1, MR3############")
# compute_inter_group_sim(mr1, 3, grouped_vecs)

# print("######### MR1############")

# sim = compute_avg_sim(mr1)
# print(sim)
