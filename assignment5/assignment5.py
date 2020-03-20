from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd

#Import Dataset
newsgroups_train = fetch_20newsgroups(subset='train')

#Change Dataset to DataFrame
data20 = pd.DataFrame(data=np.c_[newsgroups_train['data'],newsgroups_train['target']])

#Cleaning The Text by stopword removal and stemming operation
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, data20.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data20[0][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    #print(i)
    corpus.append(review)


# In[1]
#Use TF IDF 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = data20.iloc[:, 1].values


# In[2]
from sklearn.decomposition import PCA

pca = PCA(n_components=50)

pc = pca.fit_transform(X)


