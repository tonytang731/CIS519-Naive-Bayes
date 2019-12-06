#!/usr/bin/env python
# coding: utf-8

# In[270]:


import json
import numpy as np
#import math
#from sklearn.metrics import accuracy_score


# In[372]:


def checkAcc(prediction, y):
    count = 0
    for y_hat, y_val in zip(prediction, y):
        if y_hat == y_val:
            count += 1
    return count/len(y)


# In[281]:


def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    # TODO

    vocabulary = set()
    appeared = set()
    for docs in D:
        for words in docs:
            #only if appeared before, we add it to set
            if words in appeared:
                vocabulary.add(words)
            else:
                appeared.add(words)
        vocabulary.add('<unk>')
    return vocabulary


# In[6]:


class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        
        res = {}
        for tokens in doc:
            if tokens in vocab:
                res[tokens] = 1
            else:
                res['<unk>'] = 1
        return res
        #raise NotImplementedError


# In[304]:


class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        
        res = {}
        for tokens in doc:
            if tokens in vocab:
                if tokens in res:
                    res[tokens] += 1
                else:
                    res[tokens] = 1
            else:
                if '<unk>' in res:
                    res['<unk>'] += 1
                else:
                    res['<unk>'] = 1
        return res


# In[346]:


def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    res = {}
    for words in vocab:
        if words != '<unk>':
            res[words] = 0
    for docs in D:
        unk_appeared = False
        for tokens in set(docs): # to deduplicate/avoid double counting
            if tokens in vocab:
                res[tokens] += 1
            else:
                if not unk_appeared:
                    if '<unk>' in res:
                        res['<unk>'] += 1
                    else:
                        res['<unk>'] = 1
                    unk_appeared = True
    for keys in res:
        res[keys] = np.log(len(D)/res[keys])
    return res
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        res = {}
        for tokens in doc:
            if tokens in vocab:
                if tokens in res:
                    res[tokens] += 1
                else:
                    res[tokens] = 1
            else:
                if '<unk>' in res:
                    res['<unk>'] += 1
                else:
                    res['<unk>'] = 1            
        
        for keys in res:
            res[keys] *= self.idf.get(keys)
        return res


# In[46]:


# You should not need to edit this cell
def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X


# In[156]:


def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the inner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    # p_y
    p_y  = {}
    size = len(vocab)
    p_y[0], p_y[1] = y.count(0)/len(y), y.count(1)/len(y)
    
    # p_v_y
    p_v_y = {}
    p_v_y[1] = {}
    p_v_y[0] = {}
    
    totalCount_1 = 0
    totalCount_0 = 0
    
    for word in vocab:
        p_v_y[1][word] = 0
        p_v_y[0][word] = 0
    
    for doc, label in zip(X,y):
        for word in doc.keys():
            p_v_y[label][word] += doc[word]
            if label == 1:
                totalCount_1 += doc[word]
            else:
                totalCount_0 += doc[word]

    p_v_y[1] = {key: ((k + value) / (totalCount_1 + k*size)) for key, value in p_v_y[1].items()}
    p_v_y[0] = {key: ((k + value) / (totalCount_0 + k*size)) for key, value in p_v_y[0].items()}        
            
    return p_y, p_v_y


# In[348]:


def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    
    # TODO
    
    prediction = []
    confidence = [] # P(y|d)
    p_d = []
    p_d_y= []
    
    vocab = set(p_v_y[0])
    
    for docs in D:
        scores = []
        for label, prob in p_y.items():
            score = 0
            for word in docs:
                if word in vocab:
                    score += np.log(p_v_y[label][word])
                else:
                    score += np.log(p_v_y[label]['<unk>'])
            scores.append(score + np.log(prob))
        prediction.append(scores.index(max(scores)))
        p_d.append(np.logaddexp(scores[0],scores[1]))
        p_d_y.append(max(scores)) # p(D|y) * P(y)
    
    confidence = list(np.exp(np.array(p_d_y) - np.array(p_d)))
    return prediction, confidence

