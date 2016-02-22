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
from itertools import product, chain

import scipy as sp 
import numpy as np 
import warnings

#shuts up those infinite warnings coming to stdout to make it human
warnings.filterwarnings('ignore')
import sklearn
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier 
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed


from sklearn.feature_extraction.text import TfidfVectorizer


class ARModel():
    '''
    For efficiency, we use SGD, so technically this is an approximation to, or 
    variation of, the original Annotator Rationales model (which was SVM based)
    -- although the objective is the same. 
    '''
    def __init__(self, X_pos_rationales, X_neg_rationales,  
                    pos_rationales_worker_ids=None, neg_rationales_worker_ids=None, 
                    worker_qualities=None,
                    C=1, C_contrast_scalar=.1, mu=1.0, alpha=0.01, 
                    loss="log", n_jobs=1, 
                    pmids_to_rats={}, pmids_to_docs={}, pmids_to_labels={}):
        '''
        Instantiate an Annotators' rationales model.

        Parameters
        ----------
        X_pos_rationales : Vector encoding of `rationales' provided by annotators 
            for `positive' instances. Crucially, we assume that these have 
            been encoded using the same vectorizer as the X_i's. 
        X_neg_rationales : Ditto the above, for negative rationales.
        pos_rationales_worker_ids : Identifiers of the workers who provided the rationales
        worker_qualities: Basically, how much to scale contributions
        pos/neg_pmids_to_rats: dictionary of pmids to vectorized rationales for use on a per document basis
        '''
        self.X_pos_rationales = X_pos_rationales
        self.X_neg_rationales = X_neg_rationales
        self.pmids_to_rats = pmids_to_rats
        self.pmids_to_docs = pmids_to_docs
        self.pmids_to_labels = pmids_to_labels
        self.pmids_to_contrast_examples = {}
        
        self.pos_worker_ids = pos_rationales_worker_ids
        self.neg_worker_ids = neg_rationales_worker_ids
        self.worker_qualities = worker_qualities

        self.C = 1
        self.C_contrast_scalar = C_contrast_scalar
        self.mu = mu
        self.alpha = alpha

        self.loss = loss
        self.n_jobs = n_jobs
        print "loss: %s" % (self.loss)


    def cv_fit(self, X, y, alpha_vals, C_vals, C_contrast_vals, mu_vals, train_pmids, contrast_examples = 'per_document'):
        '''
        brute force (grid search) over hyper-parameters.
        '''
        best_params = np.zeros(4) # assume alpha, C, C_contrast, mu
        best_score = np.inf

        #need to do this for proper indexing later on
        train_pmids = np.array(train_pmids)

        ###
        # also keep track of the workers associated with each
        # rational instance!
        if contrast_examples =='per_document':
            self.pos_pseudo_examples, self.neg_pseudo_examples, self.pmids_to_contrast_examples = _per_document_pseudo(self, self.pmids_to_docs, self.pmids_to_rats, self.pmids_to_labels, train_pmids)
        else:
            #Dan: negative examples come from positive rationales, and vice versa. 
            self.neg_pseudo_examples, self.psuedo_neg_workers, neg_pmids_to_contrast = _generate_pseudo_examples(self, #add neg& pos, then write a little merge fxn.
                                                                    X, self.X_pos_rationales, train_pmids,  
                                                                    self.pos_worker_ids,  1)
            self.pos_pseudo_examples, self.psuedo_pos_workers, pos_pmids_to_contrast  = _generate_pseudo_examples(self,
                                                                    X, self.X_neg_rationales, train_pmids,
                                                                    self.neg_worker_ids, 1)
            self.pmids_to_contrast_examples = dict(neg_pmids_to_contrast.items(), pos_pmids_to_contrast.items())
        y = np.array(y)

        print "Initiating parallel KFolds"
        #Danbug: reducing the number of these to see what else we even do here
        result = Parallel(n_jobs=self.n_jobs, verbose=50)(delayed(parallelKFold)(self,
                                                           X,
                                                           y,
                                                           train_pmids,
                                                           cur_alpha,
                                                           C_vals[0],
                                                           C_contrast_vals[0],
                                                           mu_vals[0])
                                    #for cur_alpha, cur_C, cur_C_contrast_scalar, cur_mu
                                    #in product(alpha_vals, C_vals, C_contrast_vals, mu_vals))
                                    for cur_alpha
                                    in alpha_vals)
        print "FINISHED PARALLEL KFOLDS!!!!!"
        parameterScores = dict(result)
        best_score = min(k for k, v in parameterScores.iteritems())
        bestParams = parameterScores[best_score]
        mu_star = bestParams['mu']
        alpha_star = bestParams['alpha']
        C_star = bestParams['C']
        C_contrast_scalar_star = bestParams['C_contrast_scalar']

        print "ok -- best parameters: mu: %s; alpha: %s; C: %s, C_contrast_scalar: %s" % (
                        mu_star, alpha_star, C_star, C_contrast_scalar_star)
        print "score: %s" % best_score

        ###
        # now fit final model
        print "fitting final model to all data.."

        pdb.set_trace()
        
        instance_weights = np.ones(X.shape[0]) * C_star

        # now append pseudo instances to the training data!
        # note that we scale these by cur_mu!
        X = sp.sparse.vstack((X, self.pos_pseudo_examples/mu_star))
        y = np.hstack((y, np.ones(self.pos_pseudo_examples.shape[0])))

        X = sp.sparse.vstack((X, self.neg_pseudo_examples/mu_star))
        y = np.hstack((y, -1*np.ones(self.neg_pseudo_examples.shape[0])))
        
        total_contrastive_count = self.pos_pseudo_examples.shape[0] + self.neg_pseudo_examples.shape[0]
        contrast_weights = np.ones(total_contrastive_count) * C_star * C_contrast_scalar_star

      
        if self.worker_qualities is not None: 
            # then also scale by worker quality!
            for i in xrange(self.pos_pseudo_examples.shape[0]):
                worker_id = self.psuedo_pos_workers[i]#self.pos_worker_ids[i]
                worker_quality = self.worker_qualities[worker_id]
                contrast_weights[i] = contrast_weights[i] #* (worker_quality**2)

            for i in xrange(self.neg_pseudo_examples.shape[0]):
                worker_id = self.psuedo_neg_workers[i]#self.neg_worker_ids[i]
                worker_quality = self.worker_qualities[worker_id] 
                cur_idx = self.pos_pseudo_examples.shape[0]+i
                contrast_weights[cur_idx] = contrast_weights[cur_idx] #* (worker_quality**2)
      
        

        instance_weights = np.hstack((instance_weights, contrast_weights))

        clf = SGDClassifier(class_weight="auto", loss=self.loss, random_state=42, shuffle=True, alpha=alpha_star)
        clf.fit(X, y, sample_weight=instance_weights)
        self.clf = clf


    def fit(self, X, y):
        #danbug
        print 'called clf.fit'
        '''
        Fit the annotator rationales model using the provided training 
        data + rationales. 

        NOTE there is a small problem here; 
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
        # 
        # Note that there is arguably a small(?) issue here when you instantiate
        # this and then subsequently rely on GridSearch; namely you will 
        # be using pseudo examples generated from instances that are not in 
        # the (nested) training set during the tuning, which could
        # conceivably lead to overfitting. I suppose you *should* only use those
        # rationales associated with instances in the nested train set...
        # ignoring for now.
        if self.pos_pseudo_examples is None:
            self.pos_pseudo_examples = _generate_pseudo_examples(self, X, self.X_pos_rationales, self.mu)
            self.neg_pseudo_examples = _generate_pseudo_examples(self, X, self.X_neg_rationales, self.mu)

        #pos_pseudo_examples = _generate_pseudo_examples(X, self.X_pos_rationales, self.mu)
        #neg_pseudo_examples = _generate_pseudo_examples(X, self.X_neg_rationales, self.mu)

        # standard C for non-contrastive instances
        instance_weights = np.ones(X.shape[0]) * self.C

        ###
        # now append pseudo instances to the training data!
        X = sp.sparse.vstack((X, self.pos_pseudo_examples))
        y = np.hstack((y, np.ones(self.pos_pseudo_examples.shape[0])))

        X = sp.sparse.vstack((X, self.neg_pseudo_examples))
        y = np.hstack((y, -1*np.ones(self.neg_pseudo_examples.shape[0])))
        
        total_contrastive_count = self.pos_pseudo_examples.shape[0] + self.neg_pseudo_examples.shape[0]
        instance_weights = np.hstack((instance_weights, 
                                np.ones(total_contrastive_count) * self.C * self.C_contrast_scalar))

        print "all finished generating contrastive instances."

        print "fitting model..."

        clf = SGDClassifier(class_weight="auto", loss=self.loss, random_state=42, shuffle=True, alpha=self.alpha)
        clf.fit(X, y, sample_weight=instance_weights)
        self.clf = clf
        print "ok. done."

    def predict(self, X):
        # just wrap up the SGD call..
        return self.clf.predict(X)

    def predict_proba(self, X):
        try:
            return self.clf.predict_proba(X)
        except:
            pdb.set_trace()

    def get_params(self, deep=True):
        #return {"alpha": self.alpha, "recursive": self.recursive}
        return {"X_pos_rationales": self.X_pos_rationales, "X_neg_rationales": self.X_neg_rationales,
                    "C": self.C, "C_contrast_scalar": self.C_contrast_scalar, 
                    "mu":self.mu, "alpha":self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 


def parallelKFold(self, X, y, train_pmids, cur_alpha, cur_C, cur_C_contrast_scalar, cur_mu):
    kf = KFold(X.shape[0], n_folds=5, random_state=42)
    scores_for_params = []
    for nested_train, nested_test in kf:

        cur_X_train = X[nested_train,:]
        #pdb.set_trace()
        cur_y_train = y[nested_train]

        cur_X_test = X[nested_test,:]
        cur_y_test = y[nested_test]

        cur_pmids = train_pmids[nested_train]

        # standard C for non-contrastive instances
        instance_weights = np.ones(cur_X_train.shape[0]) * self.C

        # now append pseudo instances to the training data!
        # note that we scale these by cur_mu!
        # TODO: now that we have our dict of pseudo instances, we can modify to only call on those in this fold. 

        # use list of pmids for lookup
        # build a list of pseudo_examples and a parallel list of labels
        # then add pseudo_examples and labels to existing data
        contrast_examples = []
        contrast_labels = []

        for val in cur_pmids:
            contrast_examples.append(self.pmids_to_contrast_examples[val]/cur_mu)
            labels = np.ones(self.pmids_to_contrast_examples[val].shape[0])*self.pmids_to_labels[val]
            #most have 2 labels b/c only 1 rat, so we're still duplicating efforts here
            contrast_labels.extend(labels)

        cur_X_train = sp.sparse.vstack((cur_X_train, sp.sparse.vstack(contrast_examples)))
        cur_y_train = np.hstack((cur_y_train, contrast_labels))

        total_contrastive_count = len(contrast_labels)

        #cur_instance_weights = np.hstack((instance_weights,
        #                        np.ones(total_contrastive_count) * cur_C * cur_C_contrast_scalar))
        contrast_weights = np.ones(total_contrastive_count) * cur_C * cur_C_contrast_scalar

        #!!!!!*****WARNING: this probably does not work now that we do per document CV. Have not fixed yet. 
        if self.worker_qualities is not None:
            # then also scale by worker quality!
            for i in xrange(self.pos_pseudo_examples.shape[0]):
                worker_id = self.psuedo_pos_workers[i]#self.pos_worker_ids[i]
                worker_quality = self.worker_qualities[worker_id]
                contrast_weights[i] = contrast_weights[i] #* (worker_quality**2)

            for i in xrange(self.neg_pseudo_examples.shape[0]):
                worker_id = self.psuedo_neg_workers[i]#self.neg_worker_ids[i]
                worker_quality = self.worker_qualities[worker_id]
                cur_idx = self.pos_pseudo_examples.shape[0]+i
                contrast_weights[cur_idx] = contrast_weights[cur_idx] #* (worker_quality**2)

        #list of weights
        cur_instance_weights = np.hstack((instance_weights, contrast_weights))

        clf = SGDClassifier(class_weight="auto", loss=self.loss, random_state=42, shuffle=True, alpha=cur_alpha)
        clf.fit(cur_X_train, cur_y_train, sample_weight=cur_instance_weights)

        preds = clf.predict(cur_X_test)
        # we convert to 0/1 loss here
        errors = np.abs((1+cur_y_test)/2.0 - (1+preds)/2.0)

        # auto-set weights to equal - scales
        lambda_ = len(cur_y_test[cur_y_test<=0])/float(len(cur_y_test[cur_y_test>0]))
        #print "errors: %s; lambda: %s" % (errors, lambda_)

        errors[cur_y_test==1] = errors[cur_y_test==1]*lambda_
        #pdb.set_trace()
        cur_score = np.sum(errors)
        scores_for_params.append(cur_score)

    #key = "%s-%s-%s-%s"  % (cur_mu, cur_alpha, cur_C, cur_C_contrast_scalar)
    params = {'mu': cur_mu, 'alpha': cur_alpha, 'C': cur_C, 'C_contrast_scalar': cur_C_contrast_scalar}
    score = np.mean(scores_for_params)
    return (score, params)

def _per_document_pseudo(self, pmids_to_docs, pmids_to_rats, pmids_to_labels, train_pmids, mu=1):
    #how do we map pos/neg to appropriate rationales
    print '-- generating per document instances for %s documents --' % len(pmids_to_docs.values())
    pos_rats_masked_out = []
    neg_rats_masked_out = []
    pmids_to_contrast_examples = {}
    
    #assumes no duplicates in train_pmids
    for pmid in train_pmids:
        #store doc
        cur_doc = pmids_to_docs[pmid]
        #create master ex
        master_contrast = cur_doc.copy()
        if pmids_to_labels[pmid] == 1:
            cur_list = pos_rats_masked_out
        else:
            cur_list = neg_rats_masked_out

        for val in pmids_to_rats[pmid]:
            lil_list = []
            #loop through each row in vector
            for row in range(val.shape[0]):
                pseudoexample = cur_doc.copy()
                #mask out contrast values
                pseudoexample[0,val[row].nonzero()[1]] = 0
                lil_list.append(pseudoexample)
                #add to master
                master_contrast[0,val[row].nonzero()[1]] = 0
        lil_list.append(master_contrast)
        #ads little list to present 
        cur_list.append(sp.sparse.vstack(lil_list))
        pmids_to_contrast_examples[pmid] = sp.sparse.vstack(lil_list)
        #pmids_to_contrast_examples.append(sp.sparse.vstack(lil_list))

    #NOTE: can make a dictionary here real easy like
    neg_pseudoexamples = sp.sparse.vstack(pos_rats_masked_out)
    pos_pseudoexamples = sp.sparse.vstack(neg_rats_masked_out)
     
    return pos_pseudoexamples, neg_pseudoexamples, pmids_to_contrast_examples


def _generate_pseudo_examples(self, X, X_rationales, train_pmids, rationale_worker_ids=None, mu=1): 
    print "-- generating instances for %s rationales --" % X_rationales.shape[0]

    contrast_instances = []
    workers = []
    pmids_to_contrast_example_list = defaultdict(list)
    pmids_to_contrast_examples = {}
    #pdb.set_trace()

    ##
    # iterate over training data, figure out which instances
    # we need to add contrast examples for (i.e., which 
    # instances contain tokens in the rationales).
    results = Parallel(n_jobs=self.n_jobs,verbose=50)(delayed(_parallelPseudoExamples)(i,
                                                                                       X,
                                                                                       X_rationales,
                                                                                       train_pmids,
                                                                                       rationale_worker_ids,
                                                                                       mu)
                                                     #for i in xrange(X.shape[0]))
                                                    for i in xrange(10))
    #danbug
    for i in results:
        for ci in i[0]:
            #todo: make sure these are matrices, not rows. 
            contrast_instances.append(ci)
        for w in i[1]:
            workers.append(w)
        for ind, pmid in enumerate(i[2]):
            pmids_to_contrast_example_list[pmid].append(i[0][ind])


        #build pmid: contrast_example
        for val in pmids_to_contrast_example_list.keys():
            pmids_to_contrast_examples[val] = sp.sparse.vstack(pmids_to_contrast_example_list[val])
    
    return sp.sparse.vstack(contrast_instances), workers, pmids_to_contrast_examples


def _parallelPseudoExamples(i, X, X_rationales, train_pmids, rationale_worker_ids, mu): #train_pmids. should this be our pmids:documents dictionary? not if lined up
    contrast_instances = []
    workers = []
    pmid = train_pmids[i]
    #get all terms in document i, which is the current document that we are inducing pseudo examples upon
    X_i_nonzero = X[i].nonzero()[1]
    #for all rationales
    for j in xrange(X_rationales.shape[0]):
        #nonzero values in this rationale
        rationale_j_nonzero = X_rationales[j].nonzero()[1]
        #nonzero indices shared w/this rationale and this document
        shared_nonzero_indices = np.intersect1d(X_i_nonzero, rationale_j_nonzero)
        worker = None
        if rationale_worker_ids is not None:
            worker = rationale_worker_ids[j]

        ### TMP TMP TMP
        #if shared_nonzero_indices.shape[0] > 0:
        #pdb.set_trace()
        #if there is a match for this rationale
        if shared_nonzero_indices.shape[0] == rationale_j_nonzero.shape[0]: # experimental!
            # then introduce a contrast instance!
            # i.e., mask out rationale
            #print "ah ha!"
            pseudoexample = X[i].copy()
            pseudoexample[0,shared_nonzero_indices] = 0
            #default mu = 1, scale these later
            contrast_instances.append(pseudoexample/mu)
            workers.append(worker)
            #append pmid

    pmids = [pmid for val in range(len(contrast_instances))]
    return (contrast_instances, workers, pmids)


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

    sgd = SGDClassifier(class_weight="auto", random_state=42)
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
    kf = KFold(X.shape[0], n_folds=5, random_state=10)
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
            mu_vals = 10.0**np.arange(1,4)

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

def _safe_divide(a, b):
    if b == 0:
        return "UNDEFINED"
    return a/b

def compute_measures(tp, fp, fn, tn):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)


    sensitivity = _safe_divide(tp, tp+fn)  #tp / (tp + fn)
    specificity = _safe_divide(tn, tn+fp)  #tn / (tn + fp)
    precision   = _safe_divide(tp, tp+fp)  #tp / (tp + fp)
    #pdb.set_trace()
    #fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)
    #
    #fmeasure = 2 * (precision * sensitivity) / (precision + sensitivity)
    f2measure = "UNDEFINED"
    try: 
        f2measure = ((1+2**2) * (precision * sensitivity)) / ((2**2 * precision) + sensitivity)
    except: 
        pass 
    #f2measure = _safe_divide((1+2**2) * (precision * sensitivity), 
    #                         (2**2 * precision) + sensitivity)
    #f2measure1 = (1+2**2) * (precision * sensitivity) / ((2**2 * precision) + sensitivity)
    #pdb.set_trace()
    return sensitivity, specificity, precision, f2measure


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


