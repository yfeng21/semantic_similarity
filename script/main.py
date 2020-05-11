from __future__ import print_function, absolute_import, division, unicode_literals

import os
import random
import numpy as np

from options import get_arguments
# from sentence_transformers import SentenceTransformer

import nltk
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
import itertools
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine 
from sentence_transformers import SentenceTransformer
import spacy
from gensim.models import KeyedVectors
from collections import defaultdict
import csv
import re

model = SentenceTransformer("bert-base-nli-stsb-mean-tokens")
# model = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
nlp = spacy.load("en_core_web_md")





def main():
	args = get_arguments()
	print(args.cross_validate)
	# test_csv = 'testset'
	# group = read_e2e_csv(test_csv)
	# output_csv = open('cleaned_'+test_csv+'.csv', 'w')
	# csvwriter = csv.writer(output_csv)
	# csvwriter.writerow(['mr', 'ref'])
	# for key, values in group.items():
	# 	for value in values:
	# 		line = [key, value]
	# 		csvwriter.writerow(line)
	mr, sents = load_data(args.input_file)
	tokenized_sents, word2id, id2word = preprocess(sents)
	mr2sent = group_sents_by_mr(mr, sents)

	sent_index = {}
	for i, sent in enumerate(sents):
		sent_index[sent] = i 

	if args.get_sents:
		sents = mr2sent[args.mr1]
		print(sents)

	if args.in_group_sim:
		sents = mr2sent[args.mr1]
		in_group_sim = compute_in_group_sim(sents)
		print("In-group similarity for "+args.mr1+" is "+str(in_group_sim))


	negation_file = "negation_sents.csv"
	neg_data = pd.read_csv(negation_file)
	mr1 = neg_data['mr1']
	nl1 = neg_data['nl1']
	mr2 = neg_data['mr2']
	nl2 = neg_data['nl2']
	output_name = "negation_results/word2vec_negation.csv"
	output_csv = open(output_name, 'w')
	csvwriter = csv.writer(output_csv)
	csvwriter.writerow(['mr1', 'nl1', 'mr2', 'nl2', 'sentbert score'])
	for i in range(len(nl1)):
	    sent1 = nl1[i]
	    sent2 = nl2[i]
	    vec1 = encode_sents_word2vec([sent1])[0]
	    vec2 = encode_sents_word2vec([sent2])[0]
	    sim = 1 - cosine(vec1, vec2)
	    csvwriter.writerow([mr1[i], nl1[i], mr2[i], nl2[i], sim])


	# output_name = "word2vec_dev_in_group.csv"
	# output_csv = open(output_name, 'w')
	# csvwriter = csv.writer(output_csv)
	# csvwriter.writerow(['mr', 'word2vec score', 'word2vec variance', '# of sentences'])
	# for mr, sents in mr2sent.items():
	# 	if len(sents) <= 1:
	# 		continue
	# 	avg_sim, var = compute_in_group_sim_var_sentbert(sents)
	# 	# print(avg_sim, var)
	# 	csvwriter.writerow([mr, avg_sim, var, len(sents)])

	# print(mr2sent)

	# get_all_in_group_sim(mr2sent)
	# print(mr2sent.keys())
	# mr1 = "name[Alimentum], area[city centre], familyFriendly[yes]"
	# mr2 = 'name[Alimentum], area[city centre], familyFriendly[no]'
	

def read_e2e_csv(test_csv):
    groups_set = defaultdict(set)
    groups = defaultdict(list)
    count = 0
    with open(test_csv+".csv","r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            groups_set[row["mr"]].add(row["ref"])
            count += 1
    print("{} mr groups of {} nl sentences".format(len(groups_set),count))
    for g in groups_set:
        groups[g] = list(groups_set[g])
    return groups


def compute_inter_group_sim(mr1, mr2, mr2sent):

	sent_group1 = mr2sent[mr1]
	sent_group2 = mr2sent[mr2]

	sent_vec_group1 = model.encode(sent_group1)
	sent_vec_group2 = model.encode(sent_group2) 

	all_sim = 0
	count = 0

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

def encode_sents_word2vec(sents):
	sent_vecs = []

	for sent in sents:
		tokens = nlp(sent)
		vector = np.zeros(300)
		for token in tokens:
			vector += token.vector 
		vec = vector / len(tokens)
		# vec = nlp(sent).vector
		sent_vecs.append(vec)

	return sent_vecs

def compute_inter_group_sim_word2vec(mr1, mr2, mr2sent):
	sent_group1 = mr2sent[mr1]
	sent_group2 = mr2sent[mr2]

	sent_vec_group1 = encode_sents_word2vec(sent_group1)
	sent_vec_group2 = encode_sents_word2vec(sent_group2)

	all_sim = 0
	count = 0

	for i in range(len(sent_vec_group1)):
		for j in range(len(sent_vec_group2)):
			vec1 = sent_vec_group1[i]
			vec2 = sent_vec_group2[j]

			print(sent_group1[i]+'\t'+sent_group2[j]+'\t'+str(1-cosine(vec1,vec2)))
			# print(sent_group2[j])
			# print(1-cosine(vec1, vec2))

			all_sim += cosine(vec1, vec2)
			count += 1

	avg_sim = all_sim / count 
	avg_sim = 1 - avg_sim

	print(mr1 + " *** " + mr2 + " inter group similarity is "+str(avg_sim))	

def compute_in_group_sim_var_sentbert(sents):
	sent_vecs = model.encode(sents)
	all_sim = []

	for i in range(len(sent_vecs)-1):
		for j in range(i+1, len(sent_vecs)):
			vec1 = sent_vecs[i]
			vec2 = sent_vecs[j]
			sim = cosine(vec1, vec2)
			cosine_sim = 1 - sim
			all_sim.append(cosine_sim)

	avg_sim = sum(all_sim) / len(all_sim)
	var = np.var(all_sim)

	return avg_sim, var

def compute_in_group_sim(sents):
	sent_vecs = model.encode(sents)
	all_sim = 0
	count = 0

	for i in range(len(sent_vecs)-1):
		for j in range(i+1, len(sent_vecs)):
			vec1 = sent_vecs[i]
			vec2 = sent_vecs[j]
			sim = cosine(vec1, vec2)


			# print(sents[i]+'\t'+sents[j]+'\t'+str(1-sim))


			all_sim += sim 
			count += 1

	avg_sim = all_sim / count 

	return 1 - avg_sim

def compute_in_group_sim_var_word2vec(sents):
	sent_vecs = []
	for sent in sents:
		vec = nlp(sent).vector 
		sent_vecs.append(vec)

	all_sim = []

	for i in range(len(sent_vecs)-1):
		for j in range(i+1, len(sent_vecs)):
			vec1 = sent_vecs[i]
			vec2 = sent_vecs[j]
			sim = cosine(vec1, vec2)
			cosine_sim = 1 - sim

			all_sim.append(cosine_sim)

	avg_sim = sum(all_sim) / len(all_sim)
	var = np.var(all_sim)

	return avg_sim, var 


def compute_in_group_sim_word2vec(sents):
	sent_vecs = []
	for sent in sents:
		vec = nlp(sent).vector 
		sent_vecs.append(vec)

	all_sim = 0
	count = 0

	for i in range(len(sent_vecs)-1):
		for j in range(i+1, len(sent_vecs)):
			vec1 = sent_vecs[i]
			vec2 = sent_vecs[j]
			sim = cosine(vec1, vec2)

			print(sents[i]+'\t'+sents[j]+'\t'+str(1-sim))

			all_sim += sim 
			count += 1 

	avg_sim = all_sim / count 

	return 1 - avg_sim	

def get_all_in_group_sim(mr2sent):
	for mr, sents in mr2sent.items():
		in_group_sim = compute_in_group_sim(sents)
		print("In-group similarity for "+mr+" is "+str(in_group_sim))



def group_sents_by_mr(mr, sents):
	mr2sent = {}

	for i in range(len(mr)):
		if mr[i] not in mr2sent:
			sent_vector = sents[i]
			mr2sent[mr[i]] = [sent_vector]
		else:
			sent_vector = sents[i]
			mr2sent[mr[i]].append(sent_vector)

	return mr2sent



def load_data(filename):
	data = pd.read_csv(filename)
	mr = data['mr']
	sents = data['ref']
	return mr, sents 


def tokenize(sent):
	# input a sent and output a list
	tokenized_words = word_tokenize(sent)
	return tokenized_words

def preprocess(sents):
	# tokenize sentences and build dictionary
	tokenized_sents = [tokenize(sent) for sent in sents]

	all_tokens = itertools.chain.from_iterable(tokenized_sents)
	word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
	id_to_word = {idx: token for idx, token in word_to_id.items()}
	return tokenized_sents, word_to_id, id_to_word







if __name__ == '__main__':
	main()