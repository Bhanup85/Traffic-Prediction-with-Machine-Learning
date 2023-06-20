# Importing the Packages
import streamlit as sl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#Importing the Data Sets
 
# Training Data Set
df_train= pd.read_csv(r"D:\NIT 10AM\project\personal\traffic\datasets\test_set.csv")

# Testing Data set
df_test= pd.read_csv(r"D:\NIT 10AM\project\personal\traffic\datasets\test_set.csv")

# Splitting the Data Set into Training And Testing

X_train= df_train.iloc[:,[2,3,4,5]].values
y_train= df_train.iloc[:,6].values

X_test= df_test.iloc[:,[2,3,4,5]].values
y_test= df_test.iloc[:,6].values

# Now we apply Encoding for the Training and testing
from sklearn.preprocessing import LabelEncoder

labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
labelencoder_X3 = LabelEncoder()

# Apply label encoding on the features in the training data
X_train[:, 2] = labelencoder_X1.fit_transform(X_train[:, 2])
X_train[:, 3] = labelencoder_X2.fit_transform(X_train[:, 3])
X_train[:, 1] = labelencoder_X3.fit_transform(X_train[:, 1])

# Apply label encoding on the features in the testing data
X_test[:, 2] = labelencoder_X1.transform(X_test[:, 2])
X_test[:, 3] = labelencoder_X2.transform(X_test[:, 3])
X_test[:, 1] = labelencoder_X3.transform(X_test[:, 1])

# One Hot Encodding the Data


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The column indices to apply one-hot encoding
columns_to_encode = [1, 2, 3]

# Create the ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore'), columns_to_encode)], remainder='passthrough')

# Fit and transform the training data
X_train = ct.fit_transform(X_train)

# Transform the testing data
X_test = ct.transform(X_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler(with_mean=False)
X_train=sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Train the model with Random Forest

from sklearn.ensemble import RandomForestClassifier
rand_forest=RandomForestClassifier(n_estimators=1000,criterion="entropy",random_state=0)
rand_forest.fit(X_train,y_train)

# Now Predict the X_test
y_pred1=rand_forest.predict(X_test)
y_pred1

# Confusion Matrix

cm1=confusion_matrix(y_test, y_pred1)
print(cm1)

# accuracy
ac1= accuracy_score(y_test, y_pred1)
print(ac1)

#varience
varience1=rand_forest.score(X_test, y_test)
varience1

#Bias
bias1=rand_forest.score(X_train, y_train)
bias1

# Logestic Regression
from sklearn.linear_model import LogisticRegression
log_reg= LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

# Now predict the X_test
y_pred2=log_reg.predict(X_test)
y_pred2

# Confusion Matrix
cm2=confusion_matrix(y_test, y_pred2)
print(cm2)

# accuracy

ac2= accuracy_score(y_test, y_pred2)
print(ac2)

#varience
varience2=log_reg.score(X_test, y_test)
varience2

#Bias
bias2=log_reg.score(X_train, y_train)
bias2

#k Nearest Neighbour classifier
from sklearn.neighbors import KNeighborsClassifier
knn_class=KNeighborsClassifier(n_neighbors=3)
knn_class.fit(X_train,y_train)

# Now Predict the X_test

y_pred3=knn_class.predict(X_test)
y_pred3

# Confusion Matrix
cm3=confusion_matrix(y_test, y_pred3)
print(cm3)

# accuracy

ac3= accuracy_score(y_test, y_pred3)
print(ac3)

#varience
varience3=knn_class.score(X_test, y_test)
varience3

#Bias
bias3=knn_class.score(X_train, y_train)
bias3

#support Vector Classifier
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)

# Predecting the X_test Using SVC
y_pred4=svc.predict(X_test)
y_pred4

# Confusion Matrix
cm4=confusion_matrix(y_test, y_pred4)
print(cm4)

# accuracy
ac4= accuracy_score(y_test, y_pred4)
print(ac4)

#varience
varience4=svc.score(X_test, y_test)
varience4

#Bias
bias4=svc.score(X_train, y_train)
bias4

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dec_tree=DecisionTreeClassifier(criterion="gini", splitter="best",random_state=0,)
dec_tree.fit(X_train,y_train)


# Predecting the X_test Using SVC
y_pred5=dec_tree.predict(X_test)
y_pred5

# Confusion Matrix
cm5=confusion_matrix(y_test, y_pred5)
print(cm5)

# accuracy
ac5= accuracy_score(y_test, y_pred5)
print(ac5)

#varience
varience5=dec_tree.score(X_test, y_test)
varience5

#Bias
bias5=dec_tree.score(X_train, y_train)
bias5

sl.tittle('Traffic Prediction')

