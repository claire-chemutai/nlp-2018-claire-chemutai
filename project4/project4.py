
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# import NP as NP
from pylab import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:



file_to_read = "test_sentences.txt"  # file to be read

def train():
    doc = pd.read_csv(file_to_read, sep='\t', names=['review', 'sentiment'])
   
    wordset = set(stopwords.words('english'))
    v = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=wordset)

    class_categ = doc.sentiment  # positive and negative classes

    token = v.fit_transform(doc.review) 
    token_train, token_test, class_train, class_test = train_test_split(token, class_categ, random_state=40)

    # training the naive bayes classifier
    naive_train = naive_bayes.MultinomialNB()
    naive_train.fit(token_train, class_train)

    # training the logistic regression classifier
    log_train = LogisticRegression(penalty='l2', C=1)
    log_train.fit(token_train, class_train)

    print("Logistic Regression classifier accuracy with normalized data is %2.2f"
          % accuracy_score(class_test, log_train.predict(token_test)))

    print("Naive Bayes classifier accuracy with normalized data is %2.2f"
          % accuracy_score(class_test, naive_train.predict(token_test)))

    return naive_train, log_train, v


# In[3]:



def train_u():
    vu = TfidfVectorizer(use_idf=False, lowercase=False)
    doc = pd.read_csv(file_to_read, sep='\t', names=['review', 'sentiment'])

    class_categ_u = doc.sentiment  # positive and negative classes
    token_u = vu.fit_transform(doc.review)

    token_u_train, token_u_test, class_u_train, class_u_test = train_test_split(token_u, class_categ_u, random_state=40)

    # training the naive bayes classifier
    naive_train_u = naive_bayes.MultinomialNB()
    naive_train_u.fit(token_u_train, class_u_train)

    # training the logistic regression classifier
    log_train_u = LogisticRegression(penalty='l2', C=1)
    log_train_u.fit(token_u_train, class_u_train)

    print("Logistic Regression classifier accuracy with unnormalized data is %2.2f"
          % accuracy_score(class_u_test, log_train_u.predict(token_u_test)))

    print("Naive Bayes classifier accuracy with unnormalized data is %2.2f"
          % roc_auc_score(class_u_test, naive_train_u.predict(token_u_test)))

    return naive_train_u, log_train_u, vu


# In[4]:


def nb(cl, mod, test_file):
    naive_train, log_train, v = train()
    file = open(test_file, "r")
    predict_array = []  # initialize array to contain classifier results
    for line in file:
        # treating each line by putting them into an array using an inbuilt panda function
        movie_review_arr = pd.np.array([line])
        movie_vect = v.transform(movie_review_arr)
        class_placed = naive_train.predict(movie_vect)

        # putting the classification results into an array
        predict_array.append(class_placed)

    f = open("results-nb-n.txt", "w")

    # writing the results into a text file
    for item in predict_array:
        res = str(item)
        f.write(res.strip('[]') + "\n")
    f.close()


def lr(cl, mod, test_file):
    naive_train, log_train, v = train()
    file = open(test_file, "r")
    lr_predict_array = []  # initialize array to contain classifier results
    for line in file:
        # treating each line by putting them into an array using an inbuilt panda function
        movie_review_arr = pd.np.array([line])
        movie_vect = v.transform(movie_review_arr)
        class_placed = log_train.predict(movie_vect)

        # putting the classification results into an array
        lr_predict_array.append(class_placed)

    f = open("results-lr-n.txt", "w")

    # writing the results into a text file
    for item in lr_predict_array:
        res = str(item)
        f.write(res.strip('[]') + "\n")
    f.close()


# In[5]:


def nb_u(cl, mod, test_file):
    naive_train_u, log_train_u, vu = train_u()
    file = open(test_file, "r")
    predict_array_u = []  # initialize array to contain classifier results
    for line in file:
        # treating each line by putting them into an array using an inbuilt panda function
        movie_review_arr = pd.np.array([line])
        movie_vect = vu.transform(movie_review_arr)
        class_placed = naive_train_u.predict(movie_vect)

        # putting the classification results into an array
        predict_array_u.append(class_placed)

    f = open("results-nb-u.txt", "w")

    # writing the results into a text file
    for item in predict_array_u:
        res = str(item)
        f.write(res.strip('[]') + "\n")
    f.close()


def lr_u(cl, mod, test_file):
    naive_train_u, log_train_u, vu = train_u()
    file = open(test_file, "r")
    lr_predict_array_2 = []  # initialize array to contain classifier results
    for line in file:
        # treating each line by putting them into an array using an inbuilt panda function
        movie_review_arr = pd.np.array([line])
        movie_vect = vu.transform(movie_review_arr)
        class_placed = log_train_u.predict(movie_vect)

        # putting the classification results into an array
        lr_predict_array_2.append(class_placed)

    f = open("results-lr-u.txt", "w")

    # writing the results into a text file
    for item in lr_predict_array_2:
        res = str(item)
        f.write(res.strip('[]') + "\n")
    f.close()


# In[7]:


# Execute 

# accepting arguments from the command line

if sys.argv[1] == "nb" and sys.argv[2] == "n":
    cl = sys.argv[1]
    mod = sys.argv[2]
    test_file = sys.argv[3]
    print("\n")
    print("######## Naive Bayes Classifier with Normalized Data ########")
    nb(cl, mod, test_file)

elif sys.argv[1] == "nb" and sys.argv[2] == "u":
    cl = sys.argv[1]
    mod = sys.argv[2]
    test_file = sys.argv[3]
    print("\n")
    print("######## Naive Bayes Classifier Without Normalized Data ########")
    nb_u(cl, mod, test_file)

elif sys.argv[1] == "lr" and sys.argv[2] == "n":
    cl = sys.argv[1]
    mod = sys.argv[2]
    test_file = sys.argv[3]
    print("\n")
    print("######## Logistic Regression Classifier With Normalized Data ########")
    lr(cl, mod, test_file)

elif sys.argv[1] == "lr" and sys.argv[2] == "u":
    cl = sys.argv[1]
    mod = sys.argv[2]
    test_file = sys.argv[3]
    print("\n")
    print("######## Logistic Regression Classifier Without Normalized Data ########")
    lr_u(cl, mod, test_file)

