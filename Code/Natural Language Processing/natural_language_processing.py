# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Will contain all review which text is cleaned
corpus = []

for i in range(0, 1000):
    # Only keep letters from a to z
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    
    # Remove words that are useless to predict if the review is positive or negative
    # nltk.download('stopwords')
    splited_review = review.split()
    
    # Put words in the same time (ex: loved -> love)
    ps = PorterStemmer()
    # Use set(stopwords.words('english')) on big text, it's faster 
    splited_review = [ps.stem(word) for word in splited_review if not word in set(stopwords.words('english'))]
    
    # Reassemble the review
    review = ' '.join(splited_review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

# Count all words that appear in all review
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

# Precision = TP / (TP + FP)
precision = cm[0][0] / (cm[0][0] + cm[1][0])

# Recall = TP / (TP + FN)
recall = cm[0][0] / (cm[0][0] + cm[0][1])

# F1 Score = 2 * Precision * Recall / (Precision + Recall)
score = 2 * precision * recall / (precision + recall)
# Logistic Regression score = 0.7512
# Random Forest score = 0.7394
# Decision Tree score = 0.7114
# SVM Kernel score = 0.7071
# SVM score = 0.6926
# Naives Bayes score = 0.6467







