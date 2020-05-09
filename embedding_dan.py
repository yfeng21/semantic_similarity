import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

def heatmap(x_labels, y_labels, values):
    fig, ax = plt.subplots()
    im = ax.imshow(values)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, "%.2f"%values[i, j],
                           ha="center", va="center", color="w", 
fontsize=6)

    fig.tight_layout()
    plt.show()

messages_lit1 = """Alimentum is a family - friendly place in the city centre . 
In the city centre there is a family - friendly place called Alimentum . 
The city centre has a family - friendly restaurant named Alimentum . 
Alimentum city centre is family - friendly 
Alimentum is a family - friendly city centre . 
There is a family - friendly restaurant named Alimentum in the city centre ."""
messages_lit2 = """There is a place in the city centre , Alimentum , that is not family - friendly .
In the city centre there is a venue name Alimentum , this is not a family - friendly venue . 
Alimentum is not a family - friendly place , located in city centre . 
Alimentum is not a family - friendly arena and is located in the city centre . 
Alimentum is not a family - friendly place in the city centre . 
Alimentum in city centre is not a family - friendly place . """
messages_lit3 = """In city centre , Aromi is a coffee shop having Chinese cuisine with average customer rating and is family friendly .
Aromi is a family friendly coffee shop that serves Chinese food . It is located in the city centre and has an average customer rating . 
Aromi is a coffee shop that provides Chinese food . It has an average rating and it located in the city centre . It is family friendly . 
Aromi is a coffee shop providing Chinese food at average quality , it is in the city centre and is family friendly . 
Aromi is a family friendly coffee shop that provides Chinese food located in the city centre with average rating . 
Aromi coffee shop have Chines food , customer rating is average and it is family friendly . It is located in city centre . """
messages1 = messages_lit1.split("\n")
messages2 = messages_lit2.split("\n")
messages3 = messages_lit3.split("\n")

def inter_corr_sim(corr):
    score = 0
    for i in range(len(corr)):
        score += (sum(corr[i])-corr[i][i])/(len(corr)-1)
    score /= len(corr)
    print("inter group similarirty:{:.2f}%  ".format(score*100))

def intra_corr_sim(len1, corr):
  score = 0
  for i in range(len(corr)):
      if i < len1:
        score += sum(corr[i][len1:])/len1
      else:
        score += sum(corr[i][:len1])/(len(corr)-len1)
  score /= len(corr)
  print("intra group similarirty:{:.2f}%  ".format(score*100))

module_url = "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# sample text
# messages = [
# "There is a place in the city centre, Alimentum, that is not family-friendly.",
# "In the city centre there is a venue name Alimentum, this is not a family-friendly venue.",
# "Alimentum is a family-friendly place in the city centre.",
# "In the city centre there is a family-friendly place called Alimentum.",
# "In city centre, Aromi is a coffee shop having Chinese cuisine with average customer rating and is family friendly.",
#  "Aromi is a family friendly coffee shop that serves Chinese food. It is located in the city centre and has an average customer rating."
# ]
messages = messages1+messages2
x = [i for i in range(len(messages))]
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages})

    corr = np.inner(message_embeddings_, message_embeddings_)
    print(corr)
    len1 = len(messages1)
    heatmap(x[:len1], x[:len1], corr[:len1,:len1])
    heatmap(x, x, corr)
    inter_corr_sim(corr[:len1,:len1])
    intra_corr_sim(len1, corr)

for i in range(len(messages)):
  print("{}:{}".format(i, messages[i]))