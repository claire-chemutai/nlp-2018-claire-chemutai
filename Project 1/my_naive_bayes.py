
# coding: utf-8

# In[1]:


import random
import math
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import string
from os import listdir
from collections import Counter
from ast import literal_eval

# turn a doc into clean tokens
def cleanDocument(doc):
    words = doc.split()
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = [word for word in words if len(word) > 1]
    words = [word.lower() for word in words]
    return words

def loadfile(filename):
    lines = open(filename).readlines()
    dataset = list(lines)
    for i in range(len(dataset)):
        for x in dataset[i]:
            dataset[i] 
        return dataset

#separate into classes
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i].strip("\n")
        if ( vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def number_of_documents(dataset):
    numlines = []
    for line in dataset:
        numlines.append(dataset.count(line))
    return numlines

def logprior(D,C):
    D=len(D)
    Ncneg=[]
    for line in C['0']:
        Ncneg.append(C['0'].count(line))        
    Ncpos=[]
    for line in C['1']:
        Ncpos.append(C['1'].count(line))
    logn=math.log(sum(Ncneg)/D)
    logp=math.log(sum(Ncpos)/D)
    
    return (int(logn)),(int(logp))
    

def likelihood(V,C):
    vocab=len(V)
    negbigdoc=[]
    for line in C['0']:
        line=line[:-1]
        line = line.split()
        for word in line:
            negbigdoc.append(word)
            
    posbigdoc=[]
    for line in C['1']:
        line=line[:-1]
        line = line.split()
        for word in line:
            posbigdoc.append(word)
    
    negOccurence=[]
    for word in V:
        if word in negbigdoc:
            negOccurence.append(negbigdoc.count(word))
    for number in negOccurence:
        neglikelihood=math.log((number+1)/(len(negbigdoc)+vocab))
    negOccurences=sum(negOccurence)
            
    posOccurence=[]
    for word in V:
        if word in posbigdoc:
            posOccurence.append(posbigdoc.count(word))
    for number in posOccurence:
        poslikelihood=math.log((number+1)/(len(posbigdoc)+vocab))
    posOccurences=sum(posOccurence)
    return neglikelihood, poslikelihood


# get the number of all positive words
def TrainNaiveBayes(D,C):
    
    Ndoc=len(D)
    Nc=[0,1]
    
    logPrior=logprior(D,C)
    
    
    V = []
    for line in D:
        line=line[:-1]
        line = line.split()
        for word in line:
            if (word not in V):
                 V.append(word)
    V=cleanDocument(str(V))
    
    loglikelihood=likelihood(V,C)
  
    return logPrior,loglikelihood,V

def TestNaiveBayes(testdoc,loglikelihood,logPrior,C,V):
    for c in C:
        sum[c]=logPrior[c]
        for i in testdoc:
            word=testdoc[i]
            if word in V:
                sum[c]=sum[c]+loglikelihood[word,c]
    print (max(sum))

file=input("Enter your file name")
D = loadfile(file)
C=separateByClass(D)

logPrior,loglikelihood,V=TrainNaiveBayes(D,C)

print(TestNaiveBayes(file,loglikelihood,logPrior,C,V))

