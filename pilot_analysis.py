from nltk import word_tokenize
import pandas as pd 
import sklearn 

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
    for title, abstract in zip(appendicitis["title"], appendicitis["abstract"]):
        title_tokens =  word_tokenize(title)
        abstract_tokens = word_tokenize(abstract)
        cur_text = ["TITLE"+t for t in title_tokens if t not in STOP_WORDS]
        cur_text.extend(abstract_tokens)
        texts.append(cur_text)

    return texts, appendicitis["pmid"]


def read_lbls(labels_path="pilot-data/labels.csv"):
    lbls = pd.read_csv(labels_path)
    # all of these pmids were screened in at the citation level.
    lvl1_set = lbls["abstrackr_decision"]
    lvl2_set = lbls["include?"]
    return lvl1_set, lvl2_set

def get_q_rationales(data, qnum):
    pos_rationales = data[data["q%s"%qnum]=="Yes"]["q%skeys" % qnum].values
    neg_rationales = data[data["q%s"%qnum]=="No"]["q%skeys" % qnum].values
    return pos_rationales, neg_rationales

def rationales_exp():
    ##
    # basics: just load in the data + labels, vectorize
    annotations = load_pilot_annotations()
    texts, pmids = load_texts_and_pmids()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000)
    X = vectorizer.fit_transform(texts)
    # these are sets of pmids that indicate positive instances;
    # all other instances are negative. these are final, abstract
    # level decisions (i.e., aggregate over the sub-questions)

    lvl1_pmids, lvl2_pmids = read_lbls()
    for question_num in range(1,5):

        ##
        # now load in and encode the rationales
        pos_rationales, neg_rationales = get_q_rationales(annotations, question_num)
        X_pos_rationales = vectorizer.transform(pos_rationales)

        neg_rationales = _load_rationales(neg_rationales_path)
        X_neg_rationales = vectorizer.transform(neg_rationales)


'''
def process_pilot_results(annotations_path = "pilot-data/pilotresults.csv"):
    annotations = pd.read_csv("pilot-data/pilotresults.csv", delimiter="|", header=None)
    annotations.columns = HEADERS

    # for each question, assemble separate labels/rationales file
    for q in range(1,5):
        with open("qlabels")
'''