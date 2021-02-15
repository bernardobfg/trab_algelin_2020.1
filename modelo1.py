import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


iris = pd.read_csv('iris.csv', sep=',',
                   names=["sepal.length","sepal.width","petal.length","petal.width","species"])

transform = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
iris['species_num'] = iris['species'].map(transform)

predictors = iris.iloc[:,:-2]
target = iris['species_num']

"""
print('PREDICTOES')
print('-------------')
print(predictors.iloc[[10,40,60,80,100,101]])
print('--------------------------------')
print('TARGET')
print('--------------')
print(target.iloc[[10,40,60,80,100,101]])
"""
X_train, X_test, y_train, y_test = train_test_split(predictors,target, test_size=0.3)
#print("Training Data - 70%", X_train.shape, y_train.shape)
#print("Testing Data - 30%", X_test.shape, y_test.shape)

svm_model = svm.SVC(kernel='linear', C=1)
svm_fit = svm_model.fit(X_train, y_train)
svm_prediction = svm_fit.predict(X_test)

svm_metric = metrics.accuracy_score(svm_prediction, y_test)
#print("Accuracy", svm_metric)

table = pd.DataFrame(metrics.confusion_matrix(y_test, svm_prediction, labels=[0,1,2]),
             columns=['setosa_predicted', 'versicolor_predicted', 'virginia_predicted'],
             index=['setosa_original', 'versicolor_original', 'viriginica_original'])

#print(table)

print(X_train)
print(y_train)