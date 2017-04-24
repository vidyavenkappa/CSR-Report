
import re
import os
import math
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.linalg import svd
import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd

path = "./res/"
files = next(os.walk(path))[2]
words = []

for i in files:
    r = path + i
    with open(r) as f:
        words.append([word.lower() for line in f for word in line.split()])#for every file in the res folder, split the file into individual words and append to a list


word_count = [len(i) for i in words]


for i in words:
    for j in i:
        if not re.search(r'^[a-z]+$', j):  #remove all words which have numbers and keep only lower case texts
            if re.search(r'\W|\d', j):
                i.remove(j)


lmtzr = WordNetLemmatizer()
filtered = []
for i in words:
    filtered.append([lmtzr.lemmatize(word) for word in i if len(word) > 4 and word not in stopwords.words('english') and not re.search('\d|[@#$%^&*()|!-_{}:".,]', word)] )


list_all_words = set([i for j in filtered for i in j])

print(len(list_all_words))
count_matrix = {}

for i in list_all_words:
    count_matrix[i] = [j.count(i) for j in filtered]

top_words = []
for i in range(len(files)):
    top_words.append([])
    for word in count_matrix:
        if len(top_words[i]) > 100:
            break
        top_words[i].append([word, count_matrix[word][i]])

for i in top_words:
    i.sort(key=lambda x: -x[1])

for i in top_words:
    print(i)
    print('\n\n\n')

for i in count_matrix:   #TF IDF
    not_appeared = len(count_matrix[i]) - count_matrix[i].count(0)
    for j in range(len(count_matrix[i])):
        count_matrix[i][j] = (count_matrix[i][j] / word_count[j]) * math.log(len(word_count)/ not_appeared)


matrix = []

for i in count_matrix:
    matrix.append(count_matrix[i])


U,S,Vt = svd(matrix)

df = pd.DataFrame(U)
#fig = ff.create_scatterplotmatrix(df, height=800, width=800)
#py.iplot(fig, filename='Basic Scatterplot Matrix')

