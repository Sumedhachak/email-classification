from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score




class Classifiers():

	def __init__(self):
		sgd_clf = SGDClassifier(random_state=42, max_iter=100)
		mlp_clf = MLPClassifier(hidden_layer_sizes=(16,))
		tree_clf = DecisionTreeClassifier()
		forest_clf = RandomForestClassifier()
		adaboost_clf = AdaBoostClassifier()
		knn_clf = KNeighborsClassifier()
		nb_clf = GaussianNB()
		svm_clf = SVC()

		self.clf = {
			'SGD': sgd_clf,
			'MLP': mlp_clf,
			'Decision Tree': tree_clf,
			'Random Forest': forest_clf,
			'AdaBoost': adaboost_clf,
			'KNN': knn_clf,
			'NB': nb_clf,
			'SVM': svm_clf
		}
		self.y_preds = {}
		self.evaluation = {}


	def predict(self, X_train, y_train):
		# make predictions using each model defined above
		for clf_name, clf in self.clf.items():
			self.y_preds[clf_name] = cross_val_predict(clf, X_train, y_train, cv=3)
			print("{}: done!".format(clf_name))
		return self.y_preds

	def evaluate(self, y_train, y_preds):
		# evaluate classifiers
		for clf_name, y_pred in self.y_preds.items():
			self.evaluation[clf_name] = {}
			self.evaluation[clf_name]['accuracy_score'] = accuracy_score(y_train, y_pred)
			self.evaluation[clf_name]['confusion_matrix'] = confusion_matrix(y_train, y_pred)
			self.evaluation[clf_name]['precision_score'] = precision_score(y_train, y_pred)
			self.evaluation[clf_name]['recall_score']= recall_score(y_train, y_pred)
			self.evaluation[clf_name]['f1_score'] = f1_score(y_train, y_pred)
			print("{}: done!".format(clf_name))
		return self.evaluation




