'''
Implementation of Zaidan et al.'s "learning with rationales"
method, as specified in

Zaidan, Omar F., Jason Eisner, and Christine Piatko. 
"Machine learning with annotator rationales to reduce annotation cost." 
Proceedings of the NIPS* 2008 Workshop on Cost Sensitive Learning. 2008.

Author: byron wallace 
'''
import pdb
import csv 

import scipy as sp 
import numpy as np 

import sklearn
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier 
from sklearn.grid_search import GridSearchCV


from sklearn.feature_extraction.text import TfidfVectorizer


class ARModel():
    '''
    For efficiency, we use SGD, so technically this is an approximation to, or 
    variation of, the original Annotator Rationales model (which was SVM based).
    '''
    def __init__(self, X_pos_rationales, X_neg_rationales, 
                    C=1, C_contrast_scalar=.1, mu=10.0, alpha=0.01):
        '''
        Instantiate an Annotators' rationales model.

        Parameters
        ----------
        X_pos_rationales : Vector encoding of `rationales' provided by annotators 
            for `positive' instances. Crucially, we assume that these have 
            been encoded using the same vectorizer as the X_i's. 
        X_neg_rationales : Ditto the above, for negative rationales.
        '''
        self.X_pos_rationales = X_pos_rationales
        self.X_neg_rationales = X_neg_rationales

        self.C = 1
        self.C_contrast_scalar = C_contrast_scalar
        self.mu = mu
        self.alpha = alpha


    def fit(self, X, y):
        '''
        Fit the annotator rationales model using the provided training 
        data + rationales. 

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        y : The labels. 
        C : Standard C variable (i.e., empirical cost); trades off against 
            regularization.
        C_contrast_scalar : Contrastive psedudo instances will be weighted
            C_contrast_scalar times as much as C.  
        mu : the scalar term with which to divide pseudo-examples (see below Eq. 6 
            in reference paper).
        alpha_vals : Values to try for regularization term. Defaults to sklearn
            advised range (10.0**-np.arange(1,7)). 
        '''
        ####
        # generate psuedo (contrast) instances
        #
        # @TODO this is slow/inefficient, because we re-generate
        # these every single time -- it would be better, for a given
        # train/test split, to cache
        ####
        pos_pseudo_examples = _generate_pseudo_examples(X, self.X_pos_rationales, self.mu)
        neg_pseudo_examples = _generate_pseudo_examples(X, self.X_neg_rationales, self.mu)

        # standard C for non-contrastive instances
        instance_weights = np.ones(X.shape[0]) * self.C

        ###
        # now append pseudo instances to the training data!
        X = sp.sparse.vstack((X, pos_pseudo_examples))
        y = np.hstack((y, np.ones(pos_pseudo_examples.shape[0])))

        X = sp.sparse.vstack((X, neg_pseudo_examples))
        y = np.hstack((y, -1*np.ones(neg_pseudo_examples.shape[0])))
        
        total_contrastive_count = pos_pseudo_examples.shape[0] + neg_pseudo_examples.shape[0]
        instance_weights = np.hstack((instance_weights, 
                                np.ones(total_contrastive_count) * self.C * self.C_contrast_scalar))

        print "all finished generating contrastive instances."

        print "fitting model..."
        clf = SGDClassifier(class_weight="auto", shuffle=True, alpha=self.alpha)
        clf.fit(X, y, sample_weight=instance_weights)
        self.clf = clf
        print "ok. done."

    def predict(self, X):
        # just wrap up the SGD call..
        return self.clf.predict(X)

    def get_params(self, deep=True):
        #return {"alpha": self.alpha, "recursive": self.recursive}
        return {"X_pos_rationales": self.X_pos_rationales, "X_neg_rationales": self.X_neg_rationales,
                    "C": self.C, "C_contrast_scalar": self.C_contrast_scalar, 
                    "mu":self.mu, "alpha":self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 

def _generate_pseudo_examples(X, X_rationales, mu=1):
    print "-- generating instances for %s rationales --" % X_rationales.shape[0]

    contrast_instances = []

    ##
    # iterate over training data, figure out which instances
    # we need to add contrast examples for (i.e., which 
    # instances contain tokens in the rationales).
    for i in xrange(X.shape[0]):
        # I'm certain there's a better way of doing this!
        # but for now keeping it simple (and inefficient..)
        X_i_nonzero = X[i].nonzero()[1]
        for j in xrange(X_rationales.shape[0]):
            rationale_j_nonzero = X_rationales[j].nonzero()[1]
            shared_nonzero_indices = np.intersect1d(X_i_nonzero, rationale_j_nonzero)

            if shared_nonzero_indices.shape[0] > 0:
                # then introduce a contrast instance!
                # i.e., maske out rationale
                pseudoexample = X[i].copy()
                pseudoexample[0,shared_nonzero_indices] = 0

                contrast_instances.append(pseudoexample/mu)

    return sp.sparse.vstack(contrast_instances)

def _load_data(path):
    texts, labels, pmids = [], [], []
    csv_reader = csv.reader(open(path, 'rb'))
    csv_reader.next() # skip headers
    for r in csv_reader:
        pmid, label, text = r
        texts.append(text)
        labels.append(int(label))
        pmids.append(pmid)
    return texts, labels, pmids

def _get_baseline_SGD():
    params_d = {"alpha": 10.0**-np.arange(1,7)}

    sgd = SGDClassifier(class_weight="auto")
    clf = GridSearchCV(sgd, params_d, scoring='f1')
    return clf

def _load_rationales(path):
    return [x.strip() for x in open(path).readlines()]

def proton_beam_example(model="rationales",
                        data_path="sample-data/proton-beam-merged.csv", 
                        pos_rationales_path="sample-data/proton-positive-rationales.txt",
                        neg_rationales_path="sample-data/proton-negative-rationales.txt"):

    ##
    # basics: just load in the data + labels, vectorize
    texts, labels, pmids = _load_data(data_path)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    ##
    # now load in and encode the rationales
    pos_rationales = _load_rationales(pos_rationales_path)
    X_pos_rationales = vectorizer.transform(pos_rationales)
    
    neg_rationales = _load_rationales(neg_rationales_path)
    X_neg_rationales = vectorizer.transform(neg_rationales)
    
    ###
    # just create an arbitrary train/test split for demo purposes.
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

    sensitivities, specificities = [], []

    C = 1
    kf = KFold(X.shape[0], n_folds=5)
    cm = np.zeros(len(np.unique(y)) ** 2)
    for train, test in kf:
        if model == "rationales":
            clf = ARModel(X_pos_rationales, X_neg_rationales)
    
            ''' 
            @TODO Michael -- you'll want to do a broader search,
                potentially, here I am severely limiting the 
                search space. 
            '''
            # warning -- this is very, very slow!
            #alpha_vals = 10.0**-np.arange(1,6)
            #C_vals = 10.0**-np.arange(-3,0)
            #C_contrast_vals = 10.0**-np.arange(-3,0)
            #mu_vals = 10.0**-np.arange(-3,0)
           
            alpha_vals = 10.0**-np.arange(2,6)
            C_vals = 10.0**-np.arange(0,1)
            C_contrast_vals = 10.0**-np.arange(1,2)
            mu_vals = 10.0**np.arange(1,3)

            params_d = {"alpha": alpha_vals, 
                        "C":C_vals, 
                        "C_contrast_scalar":C_contrast_vals,
                        "mu":mu_vals}
            ### 
            # finally, just pass this forward to the SGD classifier.
            # @TODO not entirely sure you want to use the "auto" flag to 
            # class_weight here - this will up-weight positives, in this case!
            
            ar = ARModel(X_pos_rationales, X_neg_rationales)
            clf = GridSearchCV(ar, params_d, scoring='f1')
            clf.fit(X[train], y[train])
        else:
            clf = _get_baseline_SGD()
            clf.fit(X[train], y[train], 
                    sample_weight=np.ones(X[train].shape[0]) * C, shuffle=True)

        y_pred = clf.predict(X[test])
        cm += sklearn.metrics.confusion_matrix(y[test], y_pred).flatten()
    
    sensitivity, specificity, f = compute_measures(*cm / float(kf.n_folds))

    print "\n----"
    print "average results for model: %s" % model 
    print "sensitivity: %s" % sensitivity
    print "specificity: %s" % specificity
    # not the traditional F; we use spec instead 
    # of precision!
    print "F (sens/spec): %s" % f  
    print "----"

def compute_measures(tp, fp, fn, tn):
     sensitivity = tp / (tp + fn)
     specificity = tn / (tn + fp)

     fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)
     return sensitivity, specificity, fmeasure


if  __name__ =='__main__':
    '''
    Sample output:


    ----
    average results for model: rationales
    sensitivity: 0.977915194346
    specificity: 0.647058823529
    F (sens/spec): 0.77880464328
    ----

    ----
    average results for model: sgd
    sensitivity: 0.9941643324
    specificity: 0.468817204301
    F (sens/spec): 0.637166404687
    ----
    '''
    # first run fancier model -- this takes a bit of time
    proton_beam_example()
    # baseline 
    proton_beam_example(model="sgd")


