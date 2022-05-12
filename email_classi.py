'''Build a model to classify email as spam or ham. First, download examples of spam and
 ham from Apache SpamAssassin’s public datasets and then train a model to classify email.'''

# Fetch the data

import os
import tarfile
import urllib
from classifiers import Classifiers

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()

fetch_spam_data()
#Now load a few emails:

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

len(ham_filenames)

len(spam_filenames)
##Use Python's email module to parse these emails (this handles headers, encoding, and so on):

import email
import email.policy

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
#Let's look at one example of ham and one example of spam, to get a feel of what the data looks like:

print(ham_emails[1].get_content().strip())

print(spam_emails[6].get_content().strip())
#Some emails are actually multipart, with images and attachments (which can have their own attachments). Let's look at the various types of structures we have:

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

structures_counter(ham_emails).most_common()

structures_counter(spam_emails).most_common()
Now let's take a look at the email headers:

for header, value in spam_emails[0].items():
    print(header,":",value)
You need to focus on the Subject header:

spam_emails[0]["Subject"]
Now split it into a training set and a test set:

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = <<your code goes here>>(X, y, test_size=0.2, random_state=42)
let's start writing the preprocessing functions. First, we will need a function to convert HTML to plain text. The following function first drops the section, then converts all <a> tags to the word HYPERLINK, then it gets rid of all HTML tags, leaving only the plain text. For readability, it also replaces multiple newlines with single newlines, and finally it unescapes html entities (such as > or  ):

import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")

print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
Now let's write a function that takes an email as input and returns its content as plain text, whatever its format is:

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

print(email_to_text(sample_html_spam)[:100], "...")
#Now install NLTK:

#!conda install nltk

try:
    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Error: stemming requires the NLTK module.")
#    stemmer = None
#You will also need a way to replace URLs with the word "URL".

try:
    import google.colab
   # !conda install -q -U urlextract
except ImportError:
    pass

try:
    import urlextract # may require an Internet connection to download root domain names

    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None
#Now put all this together into a transformer that we will use to convert emails to word counters.

from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
X_few_wordcounts
#Now we need to convert the word count to vectors:

from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors

X_few_vectors.toarray()

vocab_transformer.vocabulary_
Now let's transform the whole dataset:

from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)
X_test_transformed = preprocess_pipeline.transform(X_test)

'''Each of the classfier will be trained using a 3-fold cross validation. 
Then, we will evaluate the confusion matrix, precision, recall and f1-score. 
Finally, we will pick the classifier with the highest recall and precision.
To achieve this, we first need to import the python scripts classifiers.py 
which contains the functions to define, train and evaluate the classifiers.'''

clfs = Classifiers()

# Let's make predictions for each model

predictions = clfs.predict(X_train_transformed.toarray(), y_train)

predictions.keys()

''' Evaluate classifiers


We now evaluate each classifier. We will look at the following performance measures:

confusion matrix: how many times a spam email is classified as spam or as ham
accuracy: how many correct classifications
precision: accuracy of correct predictions, in other words TP/(TP + FP)
recall: the ratio of correct predictions, in other words TP/(TP + FN)
f1 score: the harmonic mean of precision and recall, in other words 2/(1/precision + 1/recall)
where:

TP: No. of True positive, in our case how many spam emails are predicted as such
FP: No. of False positive, how many ham emails are predicted as spam
FN: No. of False negative, how many spam emails are predicted as ham
'''
evaluations = clfs.evaluate(y_train, predictions)

for clf_name in evaluations.keys():
    print("{}:".format(clf_name))
    print("confusion matrix: \n {}".format(evaluations[clf_name]['confusion_matrix']))
    print("accuracy: {}".format(evaluations[clf_name]['accuracy_score']))
    print("precision: {}".format(evaluations[clf_name]['precision_score']))
    print("recall: {}".format(evaluations[clf_name]['recall_score']))
    print("f1-score: {}".format(evaluations[clf_name]['f1_score']))
    print()
    
 '''
From the above data, the most promising model seems to be MLP, while the least promising seems SVM.

To select a model with a good precision/recall trade-off, we are going to look at the 
precision versus recall curve. We will also plot the threshold. '''

'''Fine-tune classifiers
Now that we have a shortlist of promising models we can fine-tune them before picking the 'best'.

To fine tune the models we will use either Grid or Randomized search (according to the model), 
and then evaluate them on the test data before choosing one.'''


# Fine tune MLP
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


mlp_best = MLPClassifier()
# param to test
mlp_param_grid = {
    'hidden_layer_sizes': [(16,), (16, 16), (32,)],
    'activation': ['relu', 'tanh', 'logistic']
}

# search the best params for adaboost
mlp_grid_search = GridSearchCV(mlp_best, mlp_param_grid, 'f1', cv=5, verbose=2)
mlp_grid_search.fit(X_train_processed, y_train)



# Fine tune AdaBoost
from sklearn.ensemble import AdaBoostClassifier

adaboost_best = AdaBoostClassifier()
# param to test
adaboost_param_grid = {
    'n_estimators': [10, 30, 50, 100]
}

# search the best params for mlp
adaboost_grid_search = GridSearchCV(adaboost_best, adaboost_param_grid, 'f1', cv=5, verbose=1)
adaboost_grid_search.fit(X_train_processed, y_train)


# Fine tune SGD
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV


sgd_best = SGDClassifier()
# param to test
sgd_param_search = {
    'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    'penalty': ['l2'],
    'n_jobs': [-1]
}

# search the best params for sgd
sgd_randomized_search = RandomizedSearchCV(sgd_best, sgd_param_search, n_iter=5, scoring='f1',
                                           cv=5, verbose=2, random_state=42)
sgd_randomized_search.fit(X_train_processed, y_train)



# Get the best estimators
mlp_best = mlp_grid_search.best_estimator_
adaboost_best = adaboost_grid_search.best_estimator_
sgd_best = sgd_randomized_search.best_estimator_
#Let's evaluate each fine tuned model on the training and test set

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
# Evaluate each model on the training set
best_clf = {
    'MLP': mlp_best,
    'AdaBoost': adaboost_best,
    'SGD': sgd_best
}

# predict
best_ypreds = {}
for clf_name, clf in best_clf.items():
    best_ypreds[clf_name] = clf.predict(X_train_processed)

# evaluate
for clf_name, y_pred in best_ypreds.items():
    print("{}:".format(clf_name))
    print("confusion matrix: \n {}".format(confusion_matrix(y_train, y_pred)))
    print("accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("precision: {}".format(precision_score(y_train, y_pred)))
    print("recall: {}".format(recall_score(y_train, y_pred)))
    print("f1-score: {}".format(f1_score(y_train, y_pred)))
    print()
    
# Evaluate each model on the test set

# predict
best_ypreds = {}
for clf_name, clf in best_clf.items():
    best_ypreds[clf_name] = clf.predict(X_test_processed)

# evaluate
for clf_name, y_pred in best_ypreds.items():
    print("{}:".format(clf_name))
    print("confusion matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("precision: {}".format(precision_score(y_test, y_pred)))
    print("recall: {}".format(recall_score(y_test, y_pred)))
    print("f1-score: {}".format(f1_score(y_test, y_pred)))
    print()
    
'''Conclusions
After fine tuning our shortlisted classifiers, we can say that the 'best' classifier is MLP with the following metrics:

accuracy: 91.5%
precision: 85.6%
recall: 87.6%
f1-score: 86.6 %
The above 'best' model can be saved for future uses:'''

import pickle

output = open('mlp_best.pkl', 'wb')
pickle.dump(mlp_best, output)
output.close()
print(mlp_best)    







































































































