import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics




for col in df:
    if not pd.api.types.is_numeric_dtype(df[col]) and col != label:
        df = pd.get_dummies(df, columns=[col])



y = df[label] # Label
X = df.drop(columns=[label]) # Features
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the labels for test dataset
y_pred = clf.predict(X_test)