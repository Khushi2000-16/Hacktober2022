#intro of Added decision tree classifier 
#Decision Tree Implementation in Python
///***its a simple representation of supervised learning technique where are data is continously slpit with help of certian parametes.
Decision tree analysis can help solve both classification & regression problems. The decision tree algorithm breaks down a dataset into smaller subsets; 
while during the same time, an associated decision tree is incrementally developed.***////

1. We import the required libraries for our decision tree analysis & pull in the required data

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
# Import train_test_split function
from sklearn import metrics
#Import scikit-learn metrics module for accuracy calculation
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

##first few rows of this dataset
pima.head()

###dependent & independent variables respectively
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

## Let’s divide the data into training & testing sets in the ratio of 70:30.
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# 70% training and 30% test

Performing The decision tree analysis using scikit learn
# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Accuracy: 0.6753246753246753



