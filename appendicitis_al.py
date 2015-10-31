import StringIO
import csv
import sys
import pdb 
import string 
import math 
import random
from collections import defaultdict 
import re
import cPickle
import os.path

import numpy as np 

from nltk import word_tokenize

import pandas as pd 

import sklearn
from sklearn import metrics
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

def load_appendicitis_annotations(annotations_path="fullscale-data/appendicitis.csv", use_grouped_data=False):
    annotations = None
    if use_grouped_data:
        annotations = pd.read_csv("fullscale-data/appendicitis_grouped.csv", delimiter="\t", header=None)
    else:
        annotations = pd.read_csv("fullscale-data/appendicitis.csv", delimiter="\t", header=None)
    annotations.columns = HEADERS
    return annotations

def load_texts_and_pmids(citations_and_labels_path="fullscale-data/appendicitis_data.csv"):
    appendicitis = pd.read_csv(citations_and_labels_path, delimiter="\t")
    texts = []
    for title, abstract in zip(appendicitis["title"].values, appendicitis["abstract"].values):
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


    return texts, appendicitis["pmid"].values


def read_lbls(labels_path="fullscale-data/appendicitis_labels.csv"):
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


def get_M_overall(annotations, train_pmids, use_grouped_data=False):
    rows_list = []

    for pmid in train_pmids:
        all_annotations_for_pmid = annotations[annotations['documentId'] == pmid]
        for worker, question_answers in all_annotations_for_pmid.groupby("workerId"):
            if use_grouped_data:
                question_answer = question_answers[['q1']].values[0]
                final_answer = 3 if ('No' in question_answer) else 4
            else:
                question_answers_txt = question_answers[['q1', 'q3', 'q4']].values[0]
                question_answer_num = question_answers[['q2']].values[0][0]
                final_answer = 3 if (
                        "No" in question_answers_txt or "\\N" in question_answers_txt or (
                        question_answer_num == '\\N' or
                        (question_answer_num != 'NoInfo' and question_answer_num < 10))) else 4

            row_d = {"workerId":worker, "label":final_answer, "documentId":pmid}
            rows_list.append(row_d)

    doc_annos = pd.DataFrame(rows_list)
    #unique_workers = list(set(doc_annos["workerId"].values))
    #pdb.set_trace()
    pivoted = doc_annos.pivot(index="documentId", columns="workerId")
    pivoted = pivoted.fillna(2)
    m = pd.DataFrame.as_matrix(pivoted)
    workers = list(pivoted['label'].keys())
    return m, workers
    

def estimate_quality_instance_level(annotations, pmids, use_grouped_data=False):
    m, workers = get_M_overall(annotations, pmids, use_grouped_data)
    instance_model = ModelB.create_initial_state(2, len(workers))
    anno = AnnotationsContainer.from_array(m, missing_values=[2])
    instance_model.map(anno.annotations) 
    proxy_skill = (instance_model.theta[:,0,0] + instance_model.theta[:,1,1]) / 2.0
    return dict(zip(workers, proxy_skill))

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
    # we use '2' as our missing value; this is later signaled 
    # to the AnnotationsContainer
    pivoted.replace(["\\N","NA"], np.nan, inplace=True)
    pivoted = pivoted.fillna(2)
    workers = list(pivoted["q%s"%qnum].keys()) # this preserves order
    # matrix 
    m = pd.DataFrame.as_matrix(pivoted["q%s"%qnum])

    return m, workers

def estimate_quality_for_q(annotations, qnum, pmids=None):
    m, workers = get_M_q(annotations, qnum, pmids=pmids)
    q_model = ModelB.create_initial_state(2, len(workers))
    #pdb.set_trace()
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

def get_SGD(class_weight="auto", loss="log", random_state=None, fit_params=None, n_jobs=1):
    #C_range = np.logspace(-2, 10, 13)
    #return SGDClassifier(penalty=None)#, class_weight="auto")
    params_d = {"alpha": 10.0**-np.arange(0,6)}
    
    q_model = SGDClassifier(class_weight=class_weight, loss=loss, random_state=random_state, n_jobs=n_jobs)

    clf = GridSearchCV(q_model, params_d, scoring='f1', fit_params=fit_params, n_jobs=n_jobs)
    return clf 

def get_svm(y, n_jobs=1):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
    clf = GridSearchCV(SVC(class_weight="auto"), param_grid=param_grid, cv=cv, scoring="f1", n_jobs=n_jobs)

    return clf


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def get_all_train_and_test_X_and_y(annotations, pmids, X_all, lvl1_pmids, lvl2_pmids, 
                use_decomposed_training=False, use_grouped_data=False, use_oracle=False):
    true_y = []  # to be used for eval
    train_y = [] # to be used for training
    worker_ids = []
    X = []
    for i, pmid in enumerate(pmids):
        ###
        #  for training (crowd labels)
        ####
        q_decisions_for_pmid = annotations[annotations['documentId'] == pmid]
        for worker, question_answers in q_decisions_for_pmid.groupby("workerId"):
            # calculate the 'effective' label given by this worker,
            # as a function of their question decisions
            if use_decomposed_training:
                if use_grouped_data:
                    raise NotImplementedError("Grouped data is not compatible with decomposed training")
                q1 = question_answers[['q1']].values[0]
                q2 = question_answers[['q2']].values[0]
                q3 = question_answers[['q3']].values[0]
                q4 = question_answers[['q4']].values[0][0]
                q1a = -1 if (q1 == "No" or q1 == "\\N") else 1
                q4a = -1 if (q4 == "No" or q4 == "\\N") else 1
                q3a = -1 if (q3 == "No" or q3 == "\\N") else 1
                q2a = -1 if (q2 == '\\N' or (q2 != 'NoInfo' and q2 < 10)) else 1

                # Extra interaction feature in the form of the final answer
                question_answers_txt = question_answers[['q1', 'q3', 'q4']].values[0]
                question_answer_num = question_answers[['q2']].values[0][0]
                final_answer = -1 if ("No" in question_answers_txt or "\\N" in question_answers_txt or
                                      (question_answer_num == '\\N' or
                                       (question_answer_num != 'NoInfo' and question_answer_num < 10)))\
                                  else 1

                # Add answers and final answer interaction feature to training data.
                train_y.append(q1a)
                train_y.append(q2a)
                train_y.append(q3a)
                train_y.append(q4a)
                train_y.append(final_answer)

                # one per question
                for _ in range(5):
                    train_worker_ids.append(worker)

                for _ in range(5):
                    X.append(X_all[i])

            else:
                final_answer = None
                if use_grouped_data:
                    # bcw 10/29 -- this seems odd to me; why are we saying '1'
                    # iff the answer to the first question is yes? @TODO revisit?
                    question_answer = question_answers[['q1']].values[0]
                    final_answer = -1 if ("No" in question_answer) else 1
                else:
                    question_answers_txt = question_answers[['q1', 'q3', 'q4']].values[0]
                    question_answer_num = question_answers[['q2']].values[0][0]
                    final_answer = -1 if ("No" in question_answers_txt or "\\N" in question_answers_txt or
                                          (question_answer_num == '\\N' or
                                           (question_answer_num != 'NoInfo' and question_answer_num < 10)))\
                                      else 1

                if use_oracle:
                    ####
                    # for evaluation; expert labels
                    ####
                    train_lbl = 1 if pmid in lvl1_pmids else -1
                    train_y.append(train_lbl)
                else:
                    train_y.append(final_answer)
                worker_ids.append(worker)

        ####
        # for evaluation; expert labels
        ####
        true_lbl = 1 if pmid in lvl1_pmids else -1
        true_y.append(true_lbl)
        
    return X, train_y, worker_ids, true_y



''' BCW (10/22/2015): factoring into routine '''
def get_train_and_test_X_y(annotations, X_all, pmids, train_pmids, test_pmids, 
                            model, lvl1_pmids, lvl2_pmids, use_grouped_data=False, use_decomposed_training=False):
    cur_train_indices, cur_test_indices = [], []

    train_y, test_y = [], []

    # Some grouped specific stuff
    train_worker_ids = [] # for grouped

    # Build train_y on questions and test_y on ground truth.
    #train_annotations = annotations.copy()
    for i, pmid in enumerate(pmids):
        if pmid in train_pmids:
            q_decisions_for_pmid = annotations[annotations['documentId'] == pmid]
            for worker, question_answers in q_decisions_for_pmid.groupby("workerId"):
                # calculate the 'effective' label given by this worker,
                # as a function of their question decisions
                if use_decomposed_training:
                    if use_grouped_data:
                        raise NotImplementedError("Grouped data is not compatible with decomposed training")
                    q1 = question_answers[['q1']].values[0]
                    q2 = question_answers[['q2']].values[0]
                    q3 = question_answers[['q3']].values[0]
                    q4 = question_answers[['q4']].values[0][0]
                    q1a = -1 if (q1 == "No" or q1 == "\\N") else 1
                    q4a = -1 if (q4 == "No" or q4 == "\\N") else 1
                    q3a = -1 if (q3 == "No" or q3 == "\\N") else 1
                    q2a = -1 if (q2 == '\\N' or (q2 != 'NoInfo' and q2 < 10)) else 1

                    # Extra interaction feature in the form of the final answer
                    question_answers_txt = question_answers[['q1', 'q3', 'q4']].values[0]
                    question_answer_num = question_answers[['q2']].values[0][0]
                    final_answer = -1 if ("No" in question_answers_txt or "\\N" in question_answers_txt or
                                          (question_answer_num == '\\N' or
                                           (question_answer_num != 'NoInfo' and question_answer_num < 10)))\
                                      else 1

                    # Add answers and final answer interaction feature to training data.
                    train_y.append(q1a)
                    train_y.append(q2a)
                    train_y.append(q3a)
                    train_y.append(q4a)
                    train_y.append(final_answer)

                    for _ in range(5):
                        cur_train_indices.append(i)

                    for _ in range(5):
                        train_worker_ids.append(worker)
                else:
                    final_answer = None
                    if use_grouped_data:
                        question_answer = question_answers[['q1']].values[0]
                        final_answer = -1 if ("No" in question_answer) else 1
                    else:
                        question_answers_txt = question_answers[['q1', 'q3', 'q4']].values[0]
                        question_answer_num = question_answers[['q2']].values[0][0]
                        final_answer = -1 if ("No" in question_answers_txt or "\\N" in question_answers_txt or
                                              (question_answer_num == '\\N' or
                                               (question_answer_num != 'NoInfo' and question_answer_num < 10)))\
                                          else 1

                    train_y.append(final_answer)
                    cur_train_indices.append(i)
                    train_worker_ids.append(worker)
        elif pmid in test_pmids:
            #np.delete(train_annotations, np.where(annotations['documentId']==pmid)[0].tolist(), 0)
            lbl = 1 if pmid in lvl1_pmids else -1
            test_y.append(lbl)
            cur_test_indices.append(i)


    X_train = X_all[cur_train_indices]
    X_test = X_all[cur_test_indices]
    test_y = np.array(test_y)
    train_y = np.array(train_y)

    return X_train, train_y, X_test, test_y, train_worker_ids


####
# bcw: reimplemented to be much simpler. arguably could use a better name.
###
def uncertainty2(model, candidate_X, pmids, batch_size):
    #pdb.set_trace()
    scores = np.abs(.5 - model.predict_proba(candidate_X)[:,0])
    indices = np.argsort(scores)[:batch_size]
    return pmids[indices]


def uncertainty(model, pooled, annotations, pmids, already_selected_indices, batch_size):
    # thus the lowest will be the closest to .5 (most uncertain)
    #pdb.set_trace()
    lim_anno_ids = [v for sublist in [np.where(annotations['documentId'] == p)[0].tolist() for p in pmids] for v in sublist]
    limited_annotations = annotations.take(lim_anno_ids,axis=0)
    corrected_indices = getAnnotations(annotations, pmids, already_selected_indices)
    scores = np.abs(.5 - model.predict_proba(pooled)[:,0])
    already_selected_mask = np.zeros(pooled.shape[0])
    already_selected_mask[corrected_indices] = 1
    scores[corrected_indices] = np.inf
    ranked_indices = np.argsort(scores)
    pmidsToLabel = []
    i = 0
    #pdb.set_trace()
    while len(pmidsToLabel) < batch_size and i < len(ranked_indices):
        pmid = limited_annotations['documentId'].tolist()[ranked_indices[i]]
        if pmid not in pmidsToLabel:
            pmidsToLabel.append(pmid)

        i = i + 1
    #pdb.set_trace()
    return pmidsToLabel

def _pretty_print_d(d): 
    for key, val in d.items(): 
        print "%s: %s" % (key, val)


def getAnnotations(annotations, train_pmids, chosen_pmids):
    ids = []
    lim_anno_ids = [v for sublist in [np.where(annotations['documentId'] == p)[0].tolist() for p in train_pmids] for v in sublist]
    limited_annotations = annotations.take(lim_anno_ids,axis=0)
    for pmid in chosen_pmids:
        ids.extend(np.where(limited_annotations['documentId']==pmid)[0].tolist())

    return ids


def corRef(reflist, pmids):
    correctedList = []
    for pmid in pmids:
        correctedList.append(np.where(reflist==pmid))

    return correctedList

def _get_mask(length, indices):
    idx = np.zeros(length, dtype='bool')
    idx[indices] = True 
    return idx


def run_AL_fp(model, al_method, batch_size, 
            num_init_labels,
            annotations, X_all, train_y, true_y, pmids, vectorizer, 
            worker_ids, use_grouped_data=False,
            use_worker_qualities=False, use_rationales=False,
            n_jobs=1, init_set_path=None):
    
    n_lbls_so_far = 0 

    # bcw: one half seems reasonable
    # michael: just trying 2/3
    total_num_lbls_to_acquire = (2*X_all.shape[0])/3.0


    # maintain the learning curve -- right now,
    # the complete confusion matrix at each step
    learning_curve = []

    # probably start by randomly selecting a few instances
    # to have labeled -- may consider explicitly 
    # starting wtih a balanced sample!
    cur_train_pmids = []
    if init_set_path is None:
        cur_train_pmids = np.random.choice(pmids, num_init_labels, replace=False).tolist()
    else:
        if os.path.isfile(init_set_path):
            cur_train_pmids = cPickle.load(open(init_set_path, 'rb'))
        else:
            cur_train_pmids = np.random.choice(pmids, num_init_labels, replace=False).tolist()
            cPickle.dump(pmids, open(init_set_path, 'wb'))

    n_lbls = num_init_labels
    

    while n_lbls < total_num_lbls_to_acquire:
       
        # once everything is labele, train the model and make predictions
        # bcw: am trusting this gives me the indices from the pmids!
        #pdb.set_trace()

        # get the indices into X for the selected PMIDs.
        cur_train_pmid_indices = [j for j in xrange(len(pmids)) if pmids[j] in cur_train_pmids]
        train_idx = _get_mask(X_all.shape[0], cur_train_pmid_indices)

        # bcw 10/29: NOTE that the None here is an argument no 
        # no longer used by the _fit_and_make_predictions routine!
        # i left because didn't want to change method signature/break
        # other things
        # @TODO remove
        #train_worker_ids = [worker_ids[j] for j in pmid_indices]
        #pdb.set_trace()
        aggregate_predictions, trained_model = _fit_and_make_predictions(
                    model, annotations, X_all, None, X_all[train_idx],
                    np.array(train_y)[train_idx], X_all[~train_idx], pmids, cur_train_pmids, 
                    vectorizer, np.array(worker_ids)[train_idx],
                    use_grouped_data=use_grouped_data, use_worker_qualities=use_worker_qualities,
                    use_rationales=use_rationales, n_jobs=n_jobs, return_model=True)

        # how are we doing so far? 
        # for future ref, we record the num lbls so 
        # far and the confusion matrix.
        if not np.array_equal(np.array(true_y)[~train_idx], aggregate_predictions):
            tn, fp, fn, tp = metrics.confusion_matrix(np.array(true_y)[~train_idx], aggregate_predictions).flatten()
        else:
            if(aggregate_predictions[0] == -1):
                tn = len(aggregate_predictions)
            else:
                tp = len(aggregate_predictions)
            fp = 0
            fn = 0
        training_lbls = np.array(true_y)[train_idx]
        # assume labels are correct
        tp += training_lbls[training_lbls>0].shape[0]
        tn += training_lbls[training_lbls<=0].shape[0]

        ###
        # bcw 10/29 TODO TODO TODO probably want to re-visit metrics. i think 
        # using total tn, fp, fn and fp counts is sensible, multipled through
        # by some scalar for asymmetry
        sensitivity, specificity, precision, f2measure = ar.compute_measures(tp, fp, fn, tn)
        #auc = metrics.roc_auc_score(np.array(true_y)[~train_idx], aggregate_predictions)
        yield_val = float(tp)/float(tp+fn)
        burden_val = float(tp+tn+fp)/float(len(pmids))
        cur_results_d = {"sensitivity":sensitivity, "specificity":specificity,
                            "precision":precision, "F2":f2measure,
                            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                            "balanced_accuracy": (sensitivity+specificity)/2.0,
                            "accuracy": metrics.accuracy_score(np.array(true_y)[~train_idx], aggregate_predictions),
                            "utility_19": float(19*yield_val+(1-burden_val))/float(19+1)}
        #"accuracy": (sensitivity*prevalence)+(specificity*(1-prevalence))

        learning_curve.append((n_lbls, cur_results_d))

        # the candidate set will depend on the method;
        # because in the case of stacked models, for example,
        # the feature sets will be different.
        candidate_set = X_all.copy()

        # now pick num_lbls_to_acquire instances still in
        # the pool using the al_method!
        # add the selected indices to the cur_train_indices
        #
        if model in ("cf-stacked", "cf-stacked-bow"):

            # in these cases, we also need the question models, so unpack 
            # these from the returned values.
            q_models, trained_model = trained_model
            #pdb.set_trace()
            # we need to make feature vectors for consumption
            # by the `stacked' model; note that we make
            # predictions for *all* instances -- including 
            # those already in the selected set!
            if model == "cf-stacked":
                q_train = np.matrix([np.array(q_m.predict_proba(X_all)[:,1]) for q_m in q_models]).T
                ##### INTERACTION FEATURE (TRAINING) #####
                train_q_fvs = np.zeros((X_all.shape[0], 1))
                train_q_fvs[:,0] = np.multiply(q_train[:,0], q_train[:,1]).T
                train_q_fvs[:,0] = np.multiply(train_q_fvs[:,0], q_train[:,2].T)
                candidate_set = np.concatenate((q_train, train_q_fvs), axis=1)
            elif model == "cf-stacked-bow":
                q_train = np.matrix([np.array(q_m.predict_proba(X_all))[:,1] for q_m in q_models]).T
                ##### INTERACTION FEATURE (TRAINING) #####
                train_q_fvs = np.zeros((X_all.shape[0], 4))
                train_q_fvs[:,0] = q_train[:,0].T
                train_q_fvs[:,1] = q_train[:,1].T
                train_q_fvs[:,2] = q_train[:,2].T
                # 3-way interaction feature, i.e. q1,q3 and q4 predictions multiplied.
                train_q_fvs[:,3] = np.multiply(q_train[:,0], q_train[:,1]).T
                train_q_fvs[:,3] = np.multiply(train_q_fvs[:,3], q_train[:,2].T)
                candidate_set = np.concatenate((candidate_set.todense(), train_q_fvs), axis=1)



        # this should be set to the set of instances in X_train
        # to `label' next. 
        to_label = None 

        ###
        if al_method == "uncertainty":
            pmids_to_label = uncertainty2(trained_model, candidate_set[~train_idx], 
                                            pmids[~train_idx], batch_size)

        elif al_method == "random":
            pmids_to_label = np.random.choice(pmids[~train_idx], batch_size, replace=False)
        else: 
            raise Exception("method not implemented!")

        # now effectively `label' the selected instances.
        cur_train_pmids.extend(pmids_to_label)
        n_lbls = len(cur_train_pmids)
        
        ### may want to update accordingly if you change
        ### batchsize?
        if (n_lbls % (10 * batch_size)) == 0:
            print "labeled %s instances so far" % n_lbls
            _pretty_print_d(cur_results_d)
            print "\n---"


    return learning_curve




def run_AL(model, al_method, batch_size, num_init_labels,
            annotations, X_all, X_train, train_y, X_test, test_y,
            pmids, train_pmids, vectorizer, train_worker_ids, use_grouped_data=False,
            use_worker_qualities=False, use_rationales=False,
            n_jobs=1, init_set_path=None):
    '''
    Run active learning given a model, al_method and train/test 
    set. 


    al_method       -- active learning strategy name (string). one of: 
                        ["uncertainty sampling", ... (to be added)]

    batch_size      -- how many examples to pick at each 'round' in al?
    num_init_labels -- how many examples to label initially
    '''

    n_lbls_so_far = 0 

    # bcw: this seems like an odd default to have... why only 1/10th
    # of the data???
    total_num_lbls_to_acquire = X_train.shape[0]/10.0


    # maintain the learning curve -- right now,
    # the complete confusion matrix at each step
    learning_curve = []

    # probably start by randomly selecting a few instances
    # to have labeled -- may consider explicitly 
    # starting wtih a balanced sample!

    
    if init_set_path is None:
        cur_train_indices = np.random.choice(train_pmids, num_init_labels, replace=False).tolist()
    else:
        if os.path.isfile(init_set_path):
            cur_train_indices = cPickle.load(open(init_set_path, 'rb'))
        else:
            cur_train_indices = np.random.choice(train_pmids, num_init_labels, replace=False).tolist()
            cPickle.dump(cur_train_indices, open(init_set_path, 'wb'))

    n_lbls = num_init_labels

    while n_lbls < total_num_lbls_to_acquire:
        #pdb.set_trace()
        # once everything is labele, train the model and make predictions
        actual_indices = getAnnotations(annotations, train_pmids, cur_train_indices)
        aggregate_predictions, trained_model = _fit_and_make_predictions(
                    model, annotations, X_all, actual_indices, X_train[actual_indices],
                    train_y[actual_indices],
                    X_test, pmids, train_pmids, vectorizer, train_worker_ids,
                    use_grouped_data=use_grouped_data, use_worker_qualities=use_worker_qualities,
                    use_rationales=use_rationales, n_jobs=n_jobs, return_model=True)

        # how are we doing so far? 
        # for future ref, we record the num lbls so 
        # far and the confusion matrix.
        tn, fp, fn, tp = metrics.confusion_matrix(test_y, aggregate_predictions).flatten()

        sensitivity, specificity, precision, f2measure = ar.compute_measures(tp, fp, fn, tn)
        auc = metrics.roc_auc_score(test_y, aggregate_predictions)
        cur_results_d = {"sensitivity":sensitivity, "specificity":specificity,
                            "precision":precision, "F2":f2measure, "AUC":auc}

        learning_curve.append((n_lbls, cur_results_d))

        # the candidate set will depend on the method;
        # because in the case of stacked models, for example,
        # the feature sets will be different.
        candidate_set = X_train.copy()

        # now pick num_lbls_to_acquire instances still in
        # the pool using the al_method!
        # add the selected indices to the cur_train_indices
        #
        if model in ("cf-stacked", "cf-stacked-bow"):
            # in these cases, we also need the question models, so unpack 
            # these from the returned values.
            q_models, trained_model = trained_model
            #pdb.set_trace()
            # we need to make feature vectors for consumption
            # by the `stacked' model; note that we make
            # predictions for *all* instances -- including 
            # those already in the selected set!
            if model == "cf-stacked":
                q_train = np.matrix([np.array(q_m.predict_proba(X_train)[:,1]) for q_m in q_models]).T
                ##### INTERACTION FEATURE (TRAINING) #####
                train_q_fvs = np.zeros((X_train.shape[0], 1))
                train_q_fvs[:,0] = np.multiply(q_train[:,0], q_train[:,1]).T
                train_q_fvs[:,0] = np.multiply(train_q_fvs[:,0], q_train[:,2].T)
                candidate_set = np.concatenate((q_train, train_q_fvs), axis=1)
            elif model == "cf-stacked-bow":
                q_train = np.matrix([np.array(q_m.predict_proba(X_train))[:,1] for q_m in q_models]).T
                ##### INTERACTION FEATURE (TRAINING) #####
                train_q_fvs = np.zeros((X_train.shape[0], 4))
                train_q_fvs[:,0] = q_train[:,0].T
                train_q_fvs[:,1] = q_train[:,1].T
                train_q_fvs[:,2] = q_train[:,2].T
                # 3-way interaction feature, i.e. q1,q3 and q4 predictions multiplied.
                train_q_fvs[:,3] = np.multiply(q_train[:,0], q_train[:,1]).T
                train_q_fvs[:,3] = np.multiply(train_q_fvs[:,3], q_train[:,2].T)
                candidate_set = np.concatenate((candidate_set.todense(), train_q_fvs), axis=1)



        # this should be set to the set of instances in X_train
        # to `label' next. 
        to_label = None 

        ###
        if al_method == "uncertainty":
            #pdb.set_trace()
            to_label = uncertainty(trained_model, candidate_set, annotations,
                                   train_pmids, cur_train_indices, batch_size)
        else: 
            raise Exception("method not implemented!")

        # now effectively `label' the selected instances.
        cur_train_indices.extend(to_label)
        n_lbls = len(cur_train_indices)
        
        ### may want to update accordingly if you change
        ### batchsize?
        if (n_lbls % (batch_size * 10)) == 0: 
            print "labeled %s instances so far" % n_lbls
            _pretty_print_d(cur_results_d)
            print "\n---"


    return learning_curve


### BCW note that cur_train_indices is NOT used here! 
### @TODO remove - leaving for now so as not to break...
def _fit_and_make_predictions(model, annotations, X_all, cur_train_indices, X_train, train_y, X_test, 
                                pmids, train_pmids, vectorizer, train_worker_ids, 
                                use_grouped_data=False,
                                use_worker_qualities=False, use_rationales=False,
                                n_jobs=1, return_model=False):
    ''' 
    fits the model specified by, er, model and then uses it to 
    make predictions 
    '''
    
    q_models = None  # will only be defined if cf-stacked or cf-stacked-bow

    ###
    # bcw: 10/23/15 -- switching to log-loss everywhere for AL implementation 
    #                   so we can use predict_proba for uncertainties
    ###

    if "cf" in model:
        if model == "cf-stacked":
            if use_grouped_data:
                    raise NotImplementedError("This CF method is not compatible with grouped data.")
            if use_rationales:
                q_models = _get_q_models(annotations, X_all, pmids, train_pmids,
                                        vectorizer, model=model,
                                        use_worker_qualities=use_worker_qualities,
                                        use_rationales=True,
                                        n_jobs=n_jobs)
            else:
                q_models = _get_q_models(annotations, X_all, pmids, train_pmids,
                                        vectorizer, model=model,
                                        use_worker_qualities=use_worker_qualities,
                                        use_rationales=False,
                                        n_jobs=n_jobs)

            ##### STACKING #####
            m = get_SGD(loss="log", class_weight="auto", random_state=42, n_jobs=n_jobs)

            # Training
            q_train = np.matrix([np.array(q_m.predict_proba(X_train))[:,1] for q_m in q_models]).T

            ##### INTERACTION FEATURE (TRAINING) #####
            train_q_fvs = np.zeros((X_train.shape[0], 1))
            train_q_fvs[:,0] = np.multiply(q_train[:,0], q_train[:,1]).T
            train_q_fvs[:,0] = np.multiply(train_q_fvs[:,0], q_train[:,2].T)
            train_new = np.concatenate((q_train, train_q_fvs), axis=1)

            print "fitting cf-stacked model... "
            m.fit(train_new, train_y)

            # Testing
            q_predictions = np.matrix([np.array(q_m.predict_proba(X_test)[:,1]) for q_m in q_models]).T

            ##### INTERACTION FEATURE (TESTING) #####
            test_q_fvs = np.zeros((X_test.shape[0], 1))
            test_q_fvs[:,0] = np.multiply(q_predictions[:,0], q_predictions[:,1]).T
            test_q_fvs[:,0] = np.multiply(test_q_fvs[:,0], q_predictions[:,2].T)
            test_new = np.concatenate((q_predictions, test_q_fvs), axis=1)

            aggregate_predictions = m.predict(test_new)
        elif model == "cf-stacked-bow":
            if use_grouped_data:
                raise NotImplementedError("This CF method is not compatible with grouped data.")
            if use_rationales:
                q_models = _get_q_models(annotations, X_all, pmids, train_pmids,
                                        vectorizer, model=model,
                                        use_worker_qualities=use_worker_qualities,
                                        use_rationales=True,
                                        n_jobs=n_jobs)
            else:
                q_models = _get_q_models(annotations, X_all, pmids, train_pmids,
                                        vectorizer, model=model,
                                        use_worker_qualities=use_worker_qualities,
                                        use_rationales=False,
                                        n_jobs=n_jobs)

            q_train = np.matrix([np.array(q_m.predict_proba(X_train))[:,1] for q_m in q_models]).T

            ##### INTERACTION FEATURE (TRAINING) #####
            train_q_fvs = np.zeros((X_train.shape[0], 4))
            train_q_fvs[:,0] = q_train[:,0].T
            train_q_fvs[:,1] = q_train[:,1].T
            train_q_fvs[:,2] = q_train[:,2].T
            # 3-way interaction feature, i.e. q1,q3 and q4 predictions multiplied.
            train_q_fvs[:,3] = np.multiply(q_train[:,0], q_train[:,1]).T
            train_q_fvs[:,3] = np.multiply(train_q_fvs[:,3], q_train[:,2].T)

            q_predictions = np.matrix([np.array(q_m.predict_proba(X_test)[:,1]) for q_m in q_models]).T

            ##### INTERACTION FEATURE (TESTING) #####
            test_q_fvs = np.zeros((X_test.shape[0], 4))
            test_q_fvs[:,0] = q_predictions[:,0].T
            test_q_fvs[:,1] = q_predictions[:,1].T
            test_q_fvs[:,2] = q_predictions[:,2].T
            # 3-way interaction feature, i.e. q1,q3 and q4 predictions multiplied.
            test_q_fvs[:,3] = np.multiply(q_predictions[:,0], q_predictions[:,1]).T
            test_q_fvs[:,3] = np.multiply(test_q_fvs[:,3], q_predictions[:,2].T)


            ##### STACKING #####
            m = get_SGD(loss="log", class_weight="auto", random_state=42, n_jobs=n_jobs)
            # Training
            X_train_new = np.concatenate((X_train.todense(), train_q_fvs), axis=1) # Add interaction features
            print "fitting cf-stacked-bow model... "
            m.fit(X_train_new, train_y)
            # Testing
            X_test_new = np.concatenate((X_test.todense(), test_q_fvs), axis=1) # Add interaction features
            candidate_set = X_train_new
            aggregate_predictions = m.predict(X_test_new)
        elif model == "cf-recomposed":
            if use_rationales:
                m = get_grouped_rationales_model(
                annotations, X_train, train_y, pmids,
                train_pmids, vectorizer,
                use_worker_qualities=use_worker_qualities,
                use_grouped_data=use_grouped_data,
                n_jobs=n_jobs)

                aggregate_predictions = m.predict(X_test)
            else:
                if use_worker_qualities:
                    instance_quality_d = estimate_quality_instance_level(annotations, pmids, use_grouped_data=use_grouped_data)
                    worker_weights = [instance_quality_d[w] for w in train_worker_ids]
                    m = get_SGD(loss="log", random_state=42, fit_params={"sample_weight":worker_weights}, n_jobs=n_jobs)
                    print "fitting cf-recomposed model... "
                    m.fit(X_train, train_y)
                else:
                    m = get_SGD(loss="log", random_state=42, n_jobs=n_jobs)
                    print "fitting cf-recomposed model... "
                    m.fit(X_train, train_y)

                aggregate_predictions = m.predict(X_test)
        else:
            raise NotImplementedError('No such method exists.')
    else:
        raise NotImplementedError('No such method exists.')

    if return_model:
        if model in ("cf-stacked", "cf-stacked-bow"):
            # then we also return the individual question models!
            return aggregate_predictions, (q_models, m)
        else:   
            return aggregate_predictions, m 

    return aggregate_predictions





'''
BCW notes (10/29/2015)
---
* Finite pool 

* Not clear if you want to do n_runs here or just call this n_runs times. 
'''
def rationales_exp_all_active_fp(model="cf-stacked", use_worker_qualities=False, 
                            n_jobs=1, n_runs=5, use_grouped_data=False,
                            use_decomposed_training=False, use_rationales=False,
                            al_method="uncertainty", batch_size=10, num_init_labels=400,
                            init_set_path=None, use_oracle=False):
    ##
    # basics: just load in the data + labels, vectorize
    annotations = load_appendicitis_annotations(use_grouped_data)
    lvl1_pmids, lvl2_pmids = read_lbls()

    # Data
    texts, pmids = load_texts_and_pmids()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_features=50000)
    X_all = vectorizer.fit_transform(texts)

    # Generating folds (pool/not pool splits, here)
    unique_labeled_pmids = list(set(annotations['documentId']))


    learning_curves = []

    #X_train, train_y, X_test, test_y, train_worker_ids = get_train_and_test_X_y(
    #        annotations, X_all, pmids, train_pmids, test_pmids, 
    #        model, lvl1_pmids, lvl2_pmids, use_grouped_data=use_grouped_data,
    #        use_decomposed_training=use_decomposed_training)

    X, train_y, train_worker_ids, true_y = get_all_train_and_test_X_and_y(
            annotations, pmids, X_all, lvl2_pmids, lvl2_pmids, 
            use_grouped_data=use_grouped_data, 
            use_decomposed_training=use_decomposed_training,
            use_oracle=use_oracle)

    for run in range(n_runs):    
        # now run active learning experiment over train/test split
        cur_learning_curve = run_AL_fp(model, al_method, batch_size, 
            num_init_labels,
            annotations, X_all, train_y, true_y, pmids, vectorizer, 
            train_worker_ids, use_grouped_data=use_grouped_data,
            use_worker_qualities=use_worker_qualities, use_rationales=use_rationales,
            n_jobs=n_jobs, init_set_path=init_set_path)

        learning_curves.append(cur_learning_curve)

    ####
    # this will be a list of length n_folds, where each entry is a list
    # describing the results over active learning for this strategy
    return learning_curves




'''
BCW notes (10/22/2015)
---
* In general, the train_ids will now refer to the set available 
    during pooled active learning (for which a label may be queried)

* Similarly, test_ids constitute the evaluation set; not considered 
    during learning. 


'''
def rationales_exp_all_active(model="cf-stacked", use_worker_qualities=False, 
                            n_jobs=1, n_folds=5, use_grouped_data=False,
                            use_decomposed_training=False, use_rationales=False,
                            al_method="uncertainty", batch_size=10, num_init_labels=200,
                            init_set_path=None):
    ##
    # basics: just load in the data + labels, vectorize
    annotations = load_appendicitis_annotations(use_grouped_data)
    lvl1_pmids, lvl2_pmids = read_lbls()

    # Data
    texts, pmids = load_texts_and_pmids()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_features=50000)
    X_all = vectorizer.fit_transform(texts)

    # Generating folds (pool/not pool splits, here)
    unique_labeled_pmids = list(set(annotations['documentId']))
    folds = KFold(len(unique_labeled_pmids), 
                    n_folds=n_folds, shuffle=True, random_state=42)

    learning_curves = []

    for train_indices, test_indices in folds:
        
        # Split into training and testing PMIDs
        train_pmids = np.array(unique_labeled_pmids)[train_indices].tolist()
        test_pmids  = np.array(unique_labeled_pmids)[test_indices].tolist()

        X_train, train_y, X_test, test_y, train_worker_ids = get_train_and_test_X_y(
                annotations, X_all, pmids, train_pmids, test_pmids, 
                model, lvl1_pmids, lvl2_pmids, use_grouped_data=use_grouped_data,
                use_decomposed_training=use_decomposed_training)

        # now run active learning experiment over train/test split
        cur_learning_curve = run_AL(model, al_method, batch_size, num_init_labels,
            annotations, X_all, X_train, train_y, X_test, test_y, pmids,
            train_pmids, vectorizer, train_worker_ids, use_grouped_data=use_grouped_data,
            use_worker_qualities=use_worker_qualities, use_rationales=use_rationales,
            n_jobs=n_jobs, init_set_path=init_set_path)

        learning_curves.append(cur_learning_curve)

    ####
    # this will be a list of length n_folds, where each entry is a list
    # describing the results over active learning for this strategy
    return learning_curves

def get_unique(rationales_d, worker_qualities):
    unique_ids, unique_rationales = [], []
    for rationale, workers in rationales_d.items():
        cur_qualities = [worker_qualities[w] for w in workers]
        best_worker = workers[np.argmax(cur_qualities)]
        unique_ids.append(best_worker)
        unique_rationales.append(rationale)

    return unique_ids, unique_rationales


def get_grouped_rationales_model(annotations, X_train, train_y, pmids, train_pmids, vectorizer, use_worker_qualities=True, use_grouped_data=False, n_jobs=1):
    pos_rationales_d, neg_rationales_d = defaultdict(list), defaultdict(list)
    overall_worker_quality_d = defaultdict(list)

    ### note that q2 is an integer (population size..)
    ### so will ignore for now?
    for question_num in range(1,5):
        if use_grouped_data and question_num != 1:
            pass
        elif question_num == 2:
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
                         loss="log",
                         n_jobs=n_jobs)
    print "fitting cf-recomposed model... "
    #X_train = X[train_indices]
    model.cv_fit(X_train, train_y, alpha_vals, C_vals, C_contrast_vals, mu_vals)

    return model 


def _get_q_models(annotations, X, pmids, train_pmids, vectorizer, 
                    model="cf-stacked", use_worker_qualities=True, use_rationales=False, n_jobs=1):
    q_models = []
    
    # Note we skip question 2 because it's numeric
    # TODO: Generalize this, so we have a list of numeric questions to ignore
    for question_num in [1,3,4]:
        # get worker quality estimates, which we'll use to 
        # scale the rationales
        worker_qualities = estimate_quality_for_q(annotations, 
            question_num, pmids=train_pmids)

        # recall that pmids aligns with X. 
        train_indicators = np.in1d(pmids, train_pmids)
        X_train = X[train_indicators]
        
        q_lbls, q_X_train, q_X_train_indices = [], [], []

        worker_ids = []
        # build up a labels vector for this question, just using
        # majority vote.
        for i, pmid in enumerate(train_pmids):
            cur_pmid_annotations = annotations[annotations['documentId'] == pmid]

            q_decisions_for_pmid = list(cur_pmid_annotations['q%s' % question_num].values)
            cur_workers = list(cur_pmid_annotations['workerId'].values)


            absent_votes = q_decisions_for_pmid.count("\\N") + q_decisions_for_pmid.count("")

            if absent_votes == len(q_decisions_for_pmid):
                pass 
            else: 
                for decision_index, d in enumerate(q_decisions_for_pmid):
                    if d == "\\N":
                        pass 
                    else:
                        worker_ids.append(cur_workers[decision_index])
                        q_X_train_indices.append(i)
                        if d in ("No", "no"):
                            q_lbls.append(-1)
                        else:
                            q_lbls.append(1)

        q_X_train = X_train[q_X_train_indices]

        if use_rationales:
            # TODO(byron.wallace@utexas.edu): Consider moving, or at least extending, this block
            # Specifically we have several methods which call this method, but they don't necessarily all want the
            # rationales to be incorporated on a per question basis.

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
                                 loss="log",
                                 n_jobs=n_jobs)
            print "cv fitting!!"
            q_model.cv_fit(q_X_train, q_lbls, alpha_vals, C_vals, C_contrast_vals, mu_vals)
            q_models.append(q_model)
        else:
            params_d = {"alpha": 10.0**-np.arange(1,7)}
            q_model = SGDClassifier(class_weight=None, loss="log", random_state=42, n_jobs=n_jobs)

            
            if use_worker_qualities:
                weights = [worker_qualities[w_id] for w_id in worker_ids]
                clf = GridSearchCV(q_model, params_d, scoring='f1',
                                fit_params={'sample_weight':weights}, n_jobs=n_jobs)
            else:
                clf = GridSearchCV(q_model, params_d, scoring='f1', fit_params=None, n_jobs=n_jobs)
            
            clf.fit(q_X_train, q_lbls)
            q_models.append(clf)

    return q_models
