# email-classification
Spam Classifier
Training a spam classifier to differentiate between spam and ham emails. The dataset used for training is the SpamAssassin public mail corpus which consists of a seleciton of mail messages, labelled as spam or ham.

Overview
The goal is to train a classification algorithm to differentiate between spam and ham emails.

To reach the goal, several classification models are first trained on the Apache SpamAssassin dataset. After evaluating each classifier, the 3 best performing one are fine-tuned and re-evaluated again. The 'best' classifier is then saved as .pkl file.

All the steps are contained and documented in the Jupyter Notebok Spam classifier. To lighten the amount of line of codes in the notebook, all the functions needed have been collected in different scripts - the notebook will take care to call and run such functions.

Classification models
The classifiers used are:

SGDClassifier

MLPClassifier

DecisionTreeClassifier

RandomForestClassifier

AdaBoostClassfier

KNNClassifier

NaiveBayes

(Linear) SVM

Dataset
The above classifier are trained on the SpamAssassin public mail corpus which consists of a selection of mail messages, labelled as spam or ham.

Evaluation
Each classifier is evaluated using the following performance measures:

confusion matrix

accuracy

precision

recall

f1 score
