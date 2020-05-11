from collections import Counter
import itertools
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd 
# from gensim.models import Word2Vec
import numpy as np
import gensim.models
from scipy.spatial.distance import cosine 
import spacy
import main
import csv

nlp = spacy.load("en_core_web_md")



filename = "cleaned_devset.csv"
data = pd.read_csv(filename)
mr = data['mr']
sents = data['ref']
mr2sent = main.group_sents_by_mr(mr, sents)

def tokenize_sent(sent):
    tokens = nlp(sent)
    tokenized_words = [token.text for token in tokens]
    return tokenized_words

def map_word_frequency(document):
    return Counter(itertools.chain(*document))

tokenized_sents = [tokenize_sent(sent) for sent in sents]

word_counts = map_word_frequency(tokenized_sents)



def get_sif_feature_vectors(sentence1, nlp, word_counts):
    # print(sentence1)
    sentence1 = nlp(sentence1)
    # print(sentence1)

    embedding_size = 300 # size of vectore in word embeddings
    a = 0.001

    sentence_set=[]

    for sentence in [sentence1]:

        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        # print(sentence_length)

        for token in sentence:
            # print(token)
            a_value = a / (a + word_counts[token.text]) # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, token.vector)) # vs += sif * word_vector

        vs = np.divide(vs, sentence_length) # weighted average

        sentence_set.append(vs)

    return sentence_set[0]


def encode_sents_word2vec_sif(sents):
    sent_vecs = []

    for sent in sents:
        vec = get_sif_feature_vectors(sent, nlp, word_counts)
        sent_vecs.append(vec)

    return sent_vecs


def compute_inter_group_sim_word2vec_sif(mr1, mr2, mr2sent):
    sent_group1 = mr2sent[mr1]
    sent_group2 = mr2sent[mr2]

    sent_vec_group1 = encode_sents_word2vec_sif(sent_group1)
    sent_vec_group2 = encode_sents_word2vec_sif(sent_group2)

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


def compute_in_group_sim_word2vec_sif(sents):
    sent_vecs = []
    for sent in sents:
        vec = get_sif_feature_vectors(sent, nlp, word_counts)
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

def compute_in_group_sim_var_word2vec_sif(sents):
    sent_vecs = []
    for sent in sents:
        vec = get_sif_feature_vectors(sent, nlp, word_counts)
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


tokenized_sents = [tokenize_sent(sent) for sent in sents]

word_counts = map_word_frequency(tokenized_sents)

negation_file = "negation_sents.csv"
neg_data = pd.read_csv(negation_file)
mr1 = neg_data['mr1']
nl1 = neg_data['nl1']
mr2 = neg_data['mr2']
nl2 = neg_data['nl2']
output_name = "negation_results/w2vsif_negation.csv"
output_csv = open(output_name, 'w')
csvwriter = csv.writer(output_csv)
csvwriter.writerow(['mr1', 'nl1', 'mr2', 'nl2', 'w2v sif score'])
for i in range(len(nl1)):
    if i >= 84:
        continue
    sent1 = nl1[i]
    sent2 = nl2[i]
    vec1 = get_sif_feature_vectors(sent1, nlp, word_counts)
    vec2 = get_sif_feature_vectors(sent2, nlp, word_counts)
    sim = 1 - cosine(vec1, vec2)
    csvwriter.writerow([mr1[i], nl1[i], mr2[i], nl2[i], sim])


# output_name = "w2vsif_test_in_group.csv"
# output_csv = open(output_name, 'w')
# csvwriter = csv.writer(output_csv)
# csvwriter.writerow(['mr', 'w2v sif score', 'w2v sif variance', '# of sentences'])
# for mr, sents in mr2sent.items():
#     if len(sents) <= 1:
#         continue
#     avg_sim, var = compute_in_group_sim_var_word2vec_sif(sents)
#     # print(avg_sim, var)
#     csvwriter.writerow([mr, avg_sim, var, len(sents)])

# sent1 = "Alimentum is a family-friendly place in the city centre."
# sent2 = "In the city centre there is a family-friendly place called Alimentum."

# # sent_set = get_sif_feature_vectors(sent1, nlp, word_counts)
# # print(sent_set)

# mr1 = "name[Alimentum], area[city centre], familyFriendly[yes]"
# mr2 = 'name[Alimentum], area[city centre], familyFriendly[no]'
# mr3 = "name[Aromi], eatType[coffee shop], food[Chinese], customer rating[average], area[city centre], familyFriendly[yes]"

# print("######### MR1, MR2############")
# compute_inter_group_sim_word2vec_sif(mr1, mr2, mr2sent)

# print("######### MR1, MR3############")
# compute_inter_group_sim_word2vec_sif(mr1, mr3, mr2sent)

# print("######### MR1 ############")
# sim = compute_in_group_sim_word2vec_sif(mr2sent[mr1])
# print(sim)

