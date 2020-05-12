import pandas as pd
import matplotlib.pyplot as plt

filename = 'w2vsif_test_in_group.csv'
split = 'test'
method = 'Word2vec with SIF'
title = method + ' score on ' + split + ' set'

data = pd.read_csv(filename)
print(data.keys())
print(data.keys()[1])
key = data.keys()[1]
score = data[key]

score.plot(title = title)
plt.show()