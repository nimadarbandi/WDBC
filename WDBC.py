import numpy as np
import matplotlib. pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('wdbc.data.csv')
X = dataset.iloc[:,2:32].values
Y = dataset.iloc[:, 1].values

dataset.head()
print("Cancer data set dimensions : {}".format(dataset.shape))

dataset.isnull().sum()
dataset.isna().sum()

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier
classifierKnn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKnn.fit(X_train, Y_train)

from sklearn.svm import SVC
classifierSVC = SVC(kernel = 'linear', random_state = 0)
classifierSVC.fit(X_train, Y_train)

from sklearn.svm import SVC
classifierKSVC = SVC(kernel = 'rbf', random_state = 0)
classifierKSVC.fit(X_train, Y_train)

from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, Y_train)


from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, Y_train)


from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, Y_train)

Y_predLR = classifierLR.predict(X_test)
Y_predDT = classifierDT.predict(X_test)
Y_predKnn = classifierKnn.predict(X_test)
Y_predSVC = classifierSVC.predict(X_test)
Y_predKSVC = classifierKSVC.predict(X_test)
Y_predNB = classifierNB.predict(X_test)
Y_predRF = classifierRF.predict(X_test)

from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(Y_test, Y_predLR)
cmDT = confusion_matrix(Y_test, Y_predDT)
cmKnn = confusion_matrix(Y_test, Y_predKnn)
cmSVC = confusion_matrix(Y_test, Y_predSVC)
cmKSVC = confusion_matrix(Y_test, Y_predKSVC)
cmNB = confusion_matrix(Y_test, Y_predNB)
cmRF = confusion_matrix(Y_test, Y_predRF)

print("Confusion Matrix for Logistic Regression:")
print(cmLR)
print("\nConfusion Matrix for Decision Tree:")
print(cmDT)
print("\nConfusion Matrix for K-Nearest Neighbors:")
print(cmKnn)
print("\nConfusion Matrix for Support Vector Classifier (Linear Kernel):")
print(cmSVC)
print("\nConfusion Matrix for Kernel Support Vector Classifier (Non-linear Kernel):")
print(cmKSVC)
print("\nConfusion Matrix for Naive Bayes:")
print(cmNB)
print("\nConfusion Matrix for Random Forest:")
print(cmRF)