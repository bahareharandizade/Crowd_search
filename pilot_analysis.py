import StringIO
import csv
import pdb 
import math 

import numpy as np 

from nltk import word_tokenize

import pandas as pd 


import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
import annotator_rationales as ar

STOP_WORDS = [l.replace("\n", "") for l in open("pubmed.stoplist", 'rU').readlines()]
HEADERS = ['workerId', 'experimentId', 'hitId', 'documentId', 'q1', 'q2', 'q3', 'q4', 'q1keys', 'q2keys', 'q3keys', 'q4keys', 'q1cust', 'q2cust', 'q3cust', 'q4cust', 'q1_norationales_expl', 'q2_norationales_expl', 'q3_norationales_expl', 'q4_norationales_expl', 'q1_norationales_reverse', 'q2_norationales_reverse', 'q3_norationales_reverse', 'q4_norationales_reverse', 'comments', 'honeypotId', 'honeypotPass', 'qualificationTest', 'timeUsed', 'ts']

def load_pilot_annotations(annotations_path="pilot-data/pilotresults.csv"):
    annotations = pd.read_csv("pilot-data/pilotresults.csv", delimiter="|", header=None)
    annotations.columns = HEADERS
    return annotations

def load_texts_and_pmids(citations_and_labels_path="pilot-data/appendicitis.csv"):
    appendicitis = pd.read_csv(citations_and_labels_path)
    texts = []
    for title, abstract in zip(appendicitis["title"].values, appendicitis["abstract"].values):
        # this means the title is missing (well, nan, which is a float)
        if isinstance(title, float): 
            title_tokens = []
        else:
            title_tokens =  word_tokenize(title)

        abstract_tokens = word_tokenize(abstract)
        ### 
        # not differentiating between titles and abstracts for now, 
        # or possibly ever, because this complicates the rationales
        # learning thing.

        #cur_text = ["TITLE"+t for t in title_tokens if t not in STOP_WORDS]
        cur_text = title_tokens
        cur_text.extend(abstract_tokens)

        texts.append(" ".join(cur_text))


    return texts, appendicitis["pmid"].values


def read_lbls(labels_path="pilot-data/labels.csv"):
    lbls = pd.read_csv(labels_path)
    # all of these pmids were screened in at the citation level.
    lvl1_set = lbls["abstrackr_decision"]
    lvl2_set = lbls["include?"]
    return lvl1_set, lvl2_set

# do we need pmids? because we lose them here!
def flatten_rationales(all_rationales):
    # s is something like 
    #   'describe a 24-year-old Pakistani man,"admitted twice to our hospital"'
    # here we parse this CSV string
    rationales_flat = []
    for s in all_rationales:
        rationales_flat.extend(csv.reader(StringIO.StringIO(s)).next())
        
    return rationales_flat



def get_q_rationales(data, qnum, pmids=None):
    pos_annotations_for_q = data[data["q%s"%qnum]=="Yes"]
    neg_annotations_for_q = data[data["q%s"%qnum]=="No"]

    if pmids is not None:
        # then only include those rationales associated with pmids of 
        # interest
        pos_annotations_for_q = \
            pos_annotations_for_q[pos_annotations_for_q['documentId'].isin(pmids)]

        neg_annotations_for_q = \
            neg_annotations_for_q[neg_annotations_for_q['documentId'].isin(pmids)]
        
    
    
    pos_rationales = pos_annotations_for_q["q%skeys" % qnum].values
    # collapse into a single set
    pos_rationales = flatten_rationales(pos_rationales)

    #pos_rationales = list(chain.from_iterable(pos_rationales))
    
    neg_rationales = neg_annotations_for_q["q%skeys" % qnum].values
    #neg_rationales = list(chain.from_iterable(neg_rationales))
    neg_rationales = flatten_rationales(neg_rationales)

    ### do we need these??
    # get pubmids
    #pos_pmids = pos_annotations_for_q["documentId"]
    #neg_pmids = neg_annotations_for_q["documentId"]

  
    # collapse into a single set; note that we may want to revisit how we do
    # this!
    #return pos_rationales, pos_pmids, neg_rationales, neg_pmids
    return list(set(pos_rationales)), list(set(neg_rationales))

def rationales_exp():
    ##
    # basics: just load in the data + labels, vectorize
    annotations = load_pilot_annotations()
    texts, pmids = load_texts_and_pmids()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000)
    # note that X and pmids will be aligned.
    X = vectorizer.fit_transform(texts)


    # these are sets of pmids that indicate positive instances;
    # all other instances are negative. these are final, abstract
    # level decisions (i.e., aggregate over the sub-questions)
    lvl1_pmids, lvl2_pmids = read_lbls()


    ### 
    # now generate folds
    unique_labeled_pmids = list(set(annotations['documentId']))
    folds = KFold(len(unique_labeled_pmids), n_folds=5)

    for train_indices, test_indices in folds: 
        train_pmids = np.array(unique_labeled_pmids)[train_indices].tolist()

        ''' @TODO refactor into separate method ''' 
        q_models = []
        ### note that q2 is an integer (population size..)
        ### so will ignore for now?
        for question_num in range(1,5):
            if question_num == 2:
                # @TODO something else?
                pass 
            else:
                ##
                # now load in and encode the rationales
                pos_rationales, neg_rationales = get_q_rationales(annotations, 
                        question_num, pmids=train_pmids)
                # note that this technically gives us tfidf vectors, but we only use 
                # these to look up non-zero entries anyway (otherwise tf-idf would be a
                # little weird here)
                X_pos_rationales = vectorizer.transform(pos_rationales)
                X_neg_rationales = vectorizer.transform(neg_rationales)

                # recall that pmids aligns with X. 
                train_indicators = np.in1d(pmids, train_pmids)
                X_train = X[train_indicators]
                X_test = X[np.logical_not(train_indicators)]

                q_lbls = []
                # build up a labels vector for this question, just using
                # majority vote.
                for pmid in train_pmids:
                    q_decisions_for_pmid = \
                        list(annotations[annotations['documentId'] == pmid]['q%s' % question_num].values)
                    no_votes = q_decisions_for_pmid.count("No") # all other decisions we take as yes
                    if no_votes > float(len(q_decisions_for_pmid))/2.0:
                        q_lbls.append(-1)
                    else:
                        q_lbls.append(1)

                
                # ok, build the model already
                # hyper-params first (for gridsearch)
                alpha_vals = 10.0**-np.arange(2,6)
                C_vals = 10.0**-np.arange(0,1)
                C_contrast_vals = 10.0**-np.arange(1,2)
                mu_vals = 10.0**np.arange(1,4)

                params_d = {"alpha": alpha_vals, 
                            "C":C_vals, 
                            "C_contrast_scalar":C_contrast_vals,
                            "mu":mu_vals}        

                q_model = ar.ARModel(X_pos_rationales, X_neg_rationales)
                clf = GridSearchCV(q_model, params_d, scoring='f1')
                clf.fit(X_train, q_lbls)
                pdb.set_trace()
                # annotations[annotations['documentId'].isin(train_pmids)]['q1']
'''
def process_pilot_results(annotations_path = "pilot-data/pilotresults.csv"):
    annotations = pd.read_csv("pilot-data/pilotresults.csv", delimiter="|", header=None)
    annotations.columns = HEADERS

    # for each question, assemble separate labels/rationales file
    for q in range(1,5):
        with open("qlabels")
'''