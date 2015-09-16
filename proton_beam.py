import StringIO
import csv
import pdb 
import string 
import math 
from collections import defaultdict 
import re 

import numpy as np 

from nltk import word_tokenize

import pandas as pd 

import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier 
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC

import pyanno
from pyanno.annotations import AnnotationsContainer
from pyanno.models import ModelB # see http://docs.enthought.com/uchicago-pyanno/pyanno.models.html#pyanno.modelB.ModelB

import annotator_rationales as ar


STOP_WORDS = [l.replace("\n", "") for l in open("pubmed.stoplist", 'rU').readlines()]
HEADERS = ['workerId', 'experimentId', 'hitId', 'documentId', 'q1', 'q2', 'q3', 'q4', 'q1keys', 'q2keys', 'q3keys', 'q4keys', 'q1cust', 'q2cust', 'q3cust', 'q4cust', 'q1_norationales_expl', 'q2_norationales_expl', 'q3_norationales_expl', 'q4_norationales_expl', 'q1_norationales_reverse', 'q2_norationales_reverse', 'q3_norationales_reverse', 'q4_norationales_reverse', 'comments', 'honeypotId', 'honeypotPass', 'qualificationTest', 'timeUsed', 'ts']

def load_protonbeam_annotations(annotations_path="fullscale-data/protonbeam.csv"):
    annotations = pd.read_csv("fullscale-data/protonbeam.csv", delimiter="\t", header=None)
    annotations.columns = HEADERS
    return annotations

def load_texts_and_pmids(citations_and_labels_path="fullscale-data/protonbeam_data.csv"):
    protonbeam = pd.read_csv(citations_and_labels_path)
    texts = []
    for title, abstract in zip(protonbeam["title"].values, protonbeam["abstract"].values):
        # this means the title is missing (well, nan, which is a float)
        if isinstance(title, float): 
            title_tokens = []
        else:
            title_tokens =  word_tokenize(title.decode('utf-8'))

        abstract_tokens = word_tokenize(abstract.decode('utf-8'))
        ### 
        # not differentiating between titles and abstracts for now, 
        # or possibly ever, because this complicates the rationales
        # learning thing.

        #cur_text = ["TITLE"+t for t in title_tokens if t not in STOP_WORDS]
        cur_text = title_tokens
        cur_text.extend(abstract_tokens)

        texts.append(" ".join(cur_text))


    return texts, protonbeam["pmid"].values


def read_lbls(labels_path="fullscale-data/protonbeam_labels.csv"):
    lbls = pd.read_csv(labels_path)
    # all of these pmids were screened in at the citation level.
    #lbls["abstrackr_decision"]
    lvl1_set = lbls[lbls["lvl1"].isin(["yes", "Yes"])]["PMID"].values

    lvl2_set = lbls[lbls["lvl2"].isin(["yes", "Yes"])]["PMID"].values
    return lvl1_set, lvl2_set

# do we need pmids? because we lose them here!
def flatten_rationales(all_rationales, workers):
    # s is something like 
    #   'describe a 24-year-old Pakistani man,"admitted twice to our hospital"'
    # here we parse this CSV string
    rationales_flat = []
    workers_extended = []
    for i,s in enumerate(all_rationales):
        cur_rationales = csv.reader(StringIO.StringIO(s)).next()
        rationales_flat.extend(cur_rationales)
        workers_extended.extend([workers[i]]*len(cur_rationales))

    return rationales_flat, workers_extended


def get_M_q(data, qnum, pmids=None):
    '''
    returns an |pmids| x |workers| matrix, where 
    columns are worker responses; also provides 
    list that maps worker ids to columns. 
    '''

    q_annotations = annotations = data[["q%s"%qnum, "documentId", "workerId"]]
    if pmids is not None:
        q_annotations = q_annotations[q_annotations['documentId'].isin(pmids)]

    pivoted = q_annotations.pivot(index="documentId", columns="workerId")
    '''
    we use these kind of wacky labels because the pyanno library
    seems to prefer integers...  
    '''
    #pivoted.replace("CantTell", 4, inplace=True)
    pivoted.replace(["Yes", "yes","CantTell"], 4, inplace=True)
    pivoted.replace(["No", "no"], 3, inplace=True)
    pivoted.replace(["\\N","NA"], np.nan, inplace=True)
    pivoted = pivoted.fillna(2)
    workers = list(pivoted["q%s"%qnum].keys()) # this preserves order
    # matrix 
    m = pd.DataFrame.as_matrix(pivoted["q%s"%qnum])
    return m, workers

def estimate_quality_for_q(annotations, qnum, pmids=None):
    m, workers = get_M_q(annotations, qnum, pmids=pmids)
    q_model = ModelB.create_initial_state(2, len(workers))
    anno = AnnotationsContainer.from_array(m, missing_values=[2])
    q_model.map(anno.annotations)
    '''
    pi[k] is the probability of label k
    theta[j,k,k'] is the probability that 
        annotator j reports label k' for an 
        item whose real label is k, i.e. 
        P( annotator j chooses k' | real label = k)
    '''
    # this is a simple mean of sensitivity and specificity
    # @TODO revisit? 
    proxy_skill = (q_model.theta[:,0,0] + q_model.theta[:,1,1]) / 2.0
    return dict(zip(workers, proxy_skill))


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
    pos_worker_ids = pos_annotations_for_q["workerId"].values

    def _quick_clean(s): 
        exclude = set(string.punctuation)
        s = re.sub("\d+", "", s) # scrub digits
        s = s.lower().strip()
        s = ''.join(ch for ch in s if ch not in exclude)
        return s 

    # collapse into a single set
    pos_rationales, pos_worker_ids = flatten_rationales(pos_rationales, pos_worker_ids)
    pos_rationales = [_quick_clean(pr) for pr in pos_rationales]
    #pos_rationales = list(chain.from_iterable(pos_rationales))
    
    neg_rationales = neg_annotations_for_q["q%skeys" % qnum].values
    neg_rationales = [_quick_clean(nr) for nr in neg_rationales]

    neg_worker_ids = neg_annotations_for_q["workerId"].values

    #neg_rationales = list(chain.from_iterable(neg_rationales))
    neg_rationales, neg_worker_ids = flatten_rationales(neg_rationales, neg_worker_ids)


    ### do we need these??
    # get pubmids
    #pos_pmids = pos_annotations_for_q["documentId"]
    #neg_pmids = neg_annotations_for_q["documentId"]

    # collapse into a single set; note that this is basically
    # the most naive means of combining rationales

    #pdb.set_trace()
    pos_rationales_to_worker_ids = defaultdict(list)
    for pos_rationale, pos_worker in zip(pos_rationales, pos_worker_ids):
        pos_rationales_to_worker_ids[pos_rationale].append(pos_worker)

    neg_rationales_to_worker_ids = defaultdict(list)
    for neg_rationale, neg_worker in zip(neg_rationales, neg_worker_ids):
        neg_rationales_to_worker_ids[neg_rationale].append(neg_worker)
    

    #return list(set(pos_rationales)), list(set(neg_rationales))

    # @TODO should probably roll up into an object
    #return pos_rationales, pos_worker_ids, neg_rationales, neg_worker_ids
    return pos_rationales_to_worker_ids, neg_rationales_to_worker_ids

'''
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

    # collapse into a single set; note that this is basically
    # the most naive means of combining rationales
    return list(set(pos_rationales)), list(set(neg_rationales))
'''

def get_SGD():
    #C_range = np.logspace(-2, 10, 13)
    return SGDClassifier(penalty=None, class_weight="auto")

def get_svm(y):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
    clf = GridSearchCV(SVC(class_weight="auto"), param_grid=param_grid, cv=cv, scoring="f1")

    return clf


def rationales_exp(model="ar", n_folds=5, use_worker_qualities=False):
    '''
    model options: 
        "ar"         -- annotators rationales model
        "baseline"   -- baseline model, builds separate classifiers   
                        for each question 
        "grouped"    -- builds a single model, ignores questions 
        "grouped-ar" -- builds a single model *and* uses rationales, but 
                            groups them together.

    '''
    assert model in ("ar", "baseline", "grouped", "grouped-ar")

    ##
    # basics: just load in the data + labels, vectorize
    annotations = load_protonbeam_annotations()
    texts, pmids = load_texts_and_pmids()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_features=50000)
    # note that X and pmids will be aligned.
    X = vectorizer.fit_transform(texts)


    # these are sets of pmids that indicate positive instances;
    # all other instances are negative. these are final, abstract
    # level decisions (i.e., aggregate over the sub-questions)
    lvl1_pmids, lvl2_pmids = read_lbls()

    ### 
    # now generate folds
    unique_labeled_pmids = list(set(annotations['documentId']))
    folds = KFold(len(unique_labeled_pmids), n_folds=n_folds, random_state=10)

    cm = np.zeros(4)
    for train_indices, test_indices in folds: 
        train_pmids = np.array(unique_labeled_pmids)[train_indices].tolist()
        test_pmids  = np.array(unique_labeled_pmids)[test_indices].tolist()
        train_y, test_y = [], []
        for pmid in test_pmids:
            lbl = 1 if pmid in lvl1_pmids else -1
            test_y.append(lbl)
        test_y = np.array(test_y)

        for pmid in train_pmids:
            lbl = 1 if pmid in lvl1_pmids else -1 
            train_y.append(lbl)
        train_y = np.array(train_y)

        if not "grouped" in model: 
            q_models = get_q_models(annotations, X, pmids, train_pmids, vectorizer, model=model, use_worker_qualities=use_worker_qualities)        
            q_train = np.matrix([np.array(q_m.predict_proba(X[train_indices]))[:,1] for q_m in q_models]).T
            #q_train = np.matrix([np.array(q_m.decision_function(X[train_indices])) for q_m in q_models]).T
            #m = get_svm(train_y)
            m = get_SGD()

            print "fittting stacked model... "
            m.fit(q_train, train_y)

            # so this is a matrix 3 columns of predictions; one per question
            # #of rows = # of test citations
            
            q_predictions = np.matrix([np.array(q_m.predict_proba(X[test_indices])[:,1]) for q_m in q_models]).T
            #q_predictions = np.matrix([np.array(q_m.decision_function(X[test_indices])) for q_m in q_models]).T
            aggregate_predictions = m.predict(q_predictions)

        else:
            if model == "grouped":
                # grouped model; simpler case
                m = get_SGD()
                m.fit(X[train_indices], train_y)

                aggregate_predictions = m.predict(X[test_indices])
            else: 
                # grouped *with rationales* 
                m = get_grouped_rationales_model(
                    annotations, X, train_y, pmids, train_pmids, train_indices, vectorizer, use_worker_qualities=use_worker_qualities) 
                
                aggregate_predictions = m.predict(X[test_indices])

        # stack these in a simple logistic

        #col_aggregates = np.array(np.sum(q_predictions, axis=1)>0).astype(np.integer) 
        #col_aggregates[col_aggregates<1]=-1
        #col_aggregates = col_aggregates[:,0]

        cm += sklearn.metrics.confusion_matrix(test_y, aggregate_predictions).flatten()
        


    #pdb.set_trace()
    tn, fp, fn, tp = cm / float(n_folds)

    #sensitivity, specificity, f = ar.compute_measures(*cm / float(n_folds))
    sensitivity, specificity, f= ar.compute_measures(tp, fp, fn, tn)

    #pdb.set_trace()
    print "average results for model: %s." % model 
    print "using worker quality estimates? %s" % use_worker_qualities
    print "\n----" 
    print "raw cm %s" % cm
    print "average cm: \n"

    #print "tp, fp, fn, tn"
    print "tn, fp, fn, tp"
    print cm/float(n_folds)
    print "sensitivity: %s" % sensitivity
    print "specificity: %s" % specificity
    # not the traditional F; we use spec instead 
    # of precision!
    print "F: %s" % f  
    print "----"

def get_unique(rationales_d, worker_qualities):
    unique_ids, unique_rationales = [], []
    for rationale, workers in rationales_d.items():
        cur_qualities = [worker_qualities[w] for w in workers]
        best_worker = workers[np.argmax(cur_qualities)]
        unique_ids.append(best_worker)
        unique_rationales.append(rationale)

    return unique_ids, unique_rationales


def get_grouped_rationales_model(annotations, X, train_y, pmids, train_pmids, train_indices, vectorizer, use_worker_qualities=True):
    pos_rationales_d, neg_rationales_d = defaultdict(list), defaultdict(list)
    overall_worker_quality_d = defaultdict(list)

    ### note that q2 is an integer (population size..)
    ### so will ignore for now?
    for question_num in range(1,5):
        if question_num == 4: # This is Proton beam specific.
            # @TODO something else?
            pass 
        else:
            # get worker quality estimates, which we'll use to 
            # scale the rationales
            worker_qualities = estimate_quality_for_q(annotations, question_num, pmids=train_pmids)
            
            # average these? 
            for w, w_q in worker_qualities.items():
                overall_worker_quality_d[w].append(w_q)

            # these are now dictionaries mapping rationales to 
            # lists of workers that provided them
            pos_rationales_d_q, neg_rationales_d_q = get_q_rationales(annotations, 
                                                            question_num, pmids=train_pmids)

            pos_rationales_d.update(pos_rationales_d_q)
            neg_rationales_d.update(neg_rationales_d_q)

            
    average_worker_qualities = {}
    # take an average for workers
    for w, qualities in overall_worker_quality_d.items(): 
        average_worker_qualities[w] = np.average(qualities)


    # collapse to unique list
    pos_rationale_worker_ids, unique_pos_rationales = get_unique(pos_rationales_d, average_worker_qualities)
    neg_rationale_worker_ids, unique_neg_rationales = get_unique(neg_rationales_d, average_worker_qualities)

    # note that this technically gives us tfidf vectors, but we only use 
    # these to look up non-zero entries anyway (otherwise tf-idf would be a
    # little weird here)
    X_pos_rationales = vectorizer.transform(unique_pos_rationales)
    X_neg_rationales = vectorizer.transform(unique_neg_rationales)

    # ok, build the model already
    # hyper-params first (for gridsearch)
    alpha_vals = 10.0**-np.arange(1,7)
    C_vals = 10.0**-np.arange(0,4)
    C_contrast_vals = 10.0**-np.arange(1,4)
    mu_vals = 10.0**np.arange(1,4)

    params_d = {"alpha": alpha_vals, 
                "C":C_vals, 
                "C_contrast_scalar":C_contrast_vals,
                "mu":mu_vals}        


    # note that you pass in the training data here, which is a little
    # weird and deviates from the usual sklearn way of doing things,
    # because this makes generating and keeping the rationales around
    # much more efficient
    if not use_worker_qualities:
        worker_qualities = None

    model = ar.ARModel(X_pos_rationales, X_neg_rationales,
                         pos_rationale_worker_ids, neg_rationale_worker_ids,
                         worker_qualities,
                         loss="log")
    print "cv fitting!!"
    X_train = X[train_indices]
    model.cv_fit(X_train, train_y, alpha_vals, C_vals, C_contrast_vals, mu_vals)

    return model 



def get_q_models(annotations, X, pmids, train_pmids, vectorizer, model="ar", use_worker_qualities=True):
    q_models = []
    ### note that q2 is an integer (population size..)
    ### so will ignore for now?
    for question_num in range(1,5):
        if question_num == 4: # This is Proton beam specific
            # @TODO something else?
            pass 
        else:
            # get worker quality estimates, which we'll use to 
            # scale the rationales
            worker_qualities = estimate_quality_for_q(annotations, question_num, pmids=train_pmids)

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

            if model == "ar":
                # annotator rationale model
                ##
                # now load in and encode the rationales
                #pos_rationales, pos_rationale_worker_ids, \
                #    neg_rationales, neg_rationale_worker_ids 

                # these are now dictionaries mapping rationales to 
                # lists of workers that provided them
                pos_rationales_d, neg_rationales_d = get_q_rationales(annotations, 
                                                                question_num, pmids=train_pmids)

                # collapse to unique list
                pos_rationale_worker_ids, unique_pos_rationales = get_unique(pos_rationales_d, worker_qualities)
                neg_rationale_worker_ids, unique_neg_rationales = get_unique(neg_rationales_d, worker_qualities)



                # note that this technically gives us tfidf vectors, but we only use 
                # these to look up non-zero entries anyway (otherwise tf-idf would be a
                # little weird here)
                X_pos_rationales = vectorizer.transform(unique_pos_rationales)
                X_neg_rationales = vectorizer.transform(unique_neg_rationales)

                
                # ok, build the model already
                # hyper-params first (for gridsearch)
                alpha_vals = 10.0**-np.arange(1,7)
                C_vals = 10.0**-np.arange(0,4)
                C_contrast_vals = 10.0**-np.arange(1,4)
                mu_vals = 10.0**np.arange(1,4)

                params_d = {"alpha": alpha_vals, 
                            "C":C_vals, 
                            "C_contrast_scalar":C_contrast_vals,
                            "mu":mu_vals}        


                # note that you pass in the training data here, which is a little
                # weird and deviates from the usual sklearn way of doing things,
                # because this makes generating and keeping the rationales around
                # much more efficient
                if not use_worker_qualities:
                    worker_qualities = None

                q_model = ar.ARModel(X_pos_rationales, X_neg_rationales,
                                     pos_rationale_worker_ids, neg_rationale_worker_ids,
                                     worker_qualities,
                                     loss="log")
                print "cv fitting!!"
                q_model.cv_fit(X_train, q_lbls, alpha_vals, C_vals, C_contrast_vals, mu_vals)
                q_models.append(q_model)
                #q_model = ar.ARModel(X_pos_rationales, X_neg_rationales, loss="log")
            else:
                # baseline 
                params_d = {"alpha": 10.0**-np.arange(1,7)}
                q_model = SGDClassifier(class_weight="auto", loss="log")

                clf = GridSearchCV(q_model, params_d, scoring='f1')
                clf.fit(X_train, q_lbls)
                q_models.append(clf)

    return q_models

                # annotations[annotations['documentId'].isin(train_pmids)]['q1']
'''
def process_pilot_results(annotations_path = "pilot-data/pilotresults.csv"):
    annotations = pd.read_csv("pilot-data/pilotresults.csv", delimiter="|", header=None)
    annotations.columns = HEADERS

    # for each question, assemble separate labels/rationales file
    for q in range(1,5):
        with open("qlabels")
'''
